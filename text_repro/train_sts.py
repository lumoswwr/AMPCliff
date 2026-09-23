import os
import json
import math
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

from AMPCliff.factory.pooling.flag_pooling import (
    FFTLatentAttentionGatePooling,
    STFTLatentAttentionGatePooling,
)


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class STSDataset(Dataset):
    def __init__(self, split, limit=None):
        self.data = split

        if limit is not None:
            limit = min(limit, len(self.data))
            self.data = self.data.select(range(limit))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        return {
            "sentence1": x["sentence1"],
            "sentence2": x["sentence2"],
            "score": float(x["score"]),
        }


class STSCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        s1 = [x["sentence1"] for x in batch]
        s2 = [x["sentence2"] for x in batch]

        score = torch.tensor(
            [x["score"] for x in batch],
            dtype=torch.float32,
        )

        # Tokenize both sides together so dynamic padding
        # uses exactly the same sequence length.
        all_sentences = s1 + s2

        tokens = self.tokenizer(
            all_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch_size = len(batch)

        tok1 = {
            k: v[:batch_size]
            for k, v in tokens.items()
        }

        tok2 = {
            k: v[batch_size:]
            for k, v in tokens.items()
        }

        return tok1, tok2, score


# ---------------------------------------------------------
# Pooling
# ---------------------------------------------------------

class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(features.dtype)

        summed = (features * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)

        return summed / denom


# ---------------------------------------------------------
# RoBERTa + Pooling
# ---------------------------------------------------------

class SentenceEncoder(nn.Module):
    def __init__(
        self,
        model_path,
        pooling,
        stft_win_length=16,
        stft_hop_length=8,
        stft_window_type="rect",
        stft_center=False,
        fixed_fft_length=None,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
        )

        d_model = self.backbone.config.hidden_size
        if pooling == "mean":
            self.pool = MeanPooling()

        elif pooling in {"FLaG", "FLaG_Hann"}:

            window_type = (
                "hann"
                if pooling == "FLaG_Hann"
                else None
            )

            self.pool = FFTLatentAttentionGatePooling(
                d_model=d_model,
                num_latents=8,
                num_heads=4,
                dropout=0.0,
                time_pool="max",
                gate_residual=True,
                eps=1e-6,
                use_gate=True,
                use_latent=True,
                post_pool_norm=True,
                window_type=window_type,
                fixed_fft_length=fixed_fft_length,
            )
        
        elif pooling in {
            "STFT_FLaG",
            "STFT_FLaG_Pos",
        }:
            self.pool = STFTLatentAttentionGatePooling(
                d_model=d_model,
                win_length=stft_win_length,
                hop_length=stft_hop_length,
                num_latents=8,
                num_heads=4,
                dropout=0.0,
                time_pool="max",
                gate_residual=True,
                eps=1e-6,
                use_gate=True,
                use_latent=True,
                post_pool_norm=True,
                use_frame_positional_encoding=(
                    pooling == "STFT_FLaG_Pos"
                ),
                max_frame_positions=32,
                local_window_type=stft_window_type,
                center_frames=stft_center,
            )

        else:
            raise ValueError(
                f"Unknown pooling: {pooling}"
            )
        
        self.pooling_name = pooling

    def encode(self, tokens):
        outputs = self.backbone(**tokens)

        hidden = outputs.last_hidden_state

        embedding = self.pool(
            hidden,
            attention_mask=tokens["attention_mask"],
        )

        return embedding

    def forward(self, tokens1, tokens2):
        # 合在一次 RoBERTa forward 中，减少额外开销
        combined = {}

        for key in tokens1:
            if key in tokens2:
                combined[key] = torch.cat(
                    [tokens1[key], tokens2[key]],
                    dim=0,
                )

        outputs = self.backbone(**combined)

        hidden = outputs.last_hidden_state

        batch_size = tokens1["input_ids"].size(0)

        hidden1 = hidden[:batch_size]
        hidden2 = hidden[batch_size:]

        mask1 = tokens1["attention_mask"]
        mask2 = tokens2["attention_mask"]

        z1 = self.pool(hidden1, mask1)
        z2 = self.pool(hidden2, mask2)

        return z1, z2


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def correlation_metrics(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    pearson = pearsonr(pred, true).statistic
    spearman = spearmanr(pred, true).statistic

    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
    }


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    preds = []
    labels = []

    for tok1, tok2, score in loader:
        tok1 = {
            k: v.to(device)
            for k, v in tok1.items()
        }

        tok2 = {
            k: v.to(device)
            for k, v in tok2.items()
        }

        z1, z2 = model(tok1, tok2)

        similarity = F.cosine_similarity(
            z1,
            z2,
            dim=-1,
        )

        preds.extend(
            similarity.detach().cpu().numpy().tolist()
        )

        labels.extend(
            score.numpy().tolist()
        )

    metrics = correlation_metrics(preds, labels)

    return metrics, preds, labels


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train(args):
    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device:", device)
    print("seed:", args.seed)
    print("pooling:", args.pooling)

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------

    raw = load_from_disk(args.data_path)

    print("\nDataset sizes:")
    for split in raw:
        print(split, len(raw[split]))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    train_ds = STSDataset(
        raw["train"],
        args.limit_train,
    )

    val_ds = STSDataset(
        raw["validation"],
        args.limit_val,
    )

    test_ds = STSDataset(
        raw["test"],
        args.limit_test,
    )

    collator = STSCollator(
        tokenizer,
        max_length=args.max_length,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = SentenceEncoder(
        args.model_path,
        args.pooling,
        stft_win_length=args.stft_win_length,
        stft_hop_length=args.stft_hop_length,
        stft_window_type=args.stft_window_type,
        stft_center=args.stft_center,
        fixed_fft_length=args.fixed_fft_length,
    ).to(device)

    print("\nBackbone hidden size:",
          model.backbone.config.hidden_size)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("total params:", total_params)
    print("trainable params:", trainable_params)

    # -----------------------------------------------------
    # Differential learning rates
    # -----------------------------------------------------

    parameter_groups = [
        {
            "params": model.backbone.parameters(),
            "lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
        }
    ]

    pool_params = [
        p for p in model.pool.parameters()
        if p.requires_grad
    ]

    if pool_params:
        parameter_groups.append({
            "params": pool_params,
            "lr": args.pool_lr,
            "weight_decay": args.weight_decay,
        })

    optimizer = torch.optim.AdamW(
        parameter_groups,
        eps=1e-8,
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / args.grad_accum
    )

    total_steps = (
        updates_per_epoch * args.epochs
    )

    warmup_steps = int(
        total_steps * args.warmup_ratio
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print("optimizer steps:", total_steps)
    print("warmup steps:", warmup_steps)

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    if args.experiment_name is None:
        run_dir = (
            Path(args.output_dir)
            / args.pooling
            / f"seed_{args.seed}"
        )
    else:
        run_dir = (
            Path(args.output_dir)
            / "experiments"
            / args.experiment_name
            / f"seed_{args.seed}"
        )
    
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = run_dir / "best_model.pt"

    config = vars(args).copy()

    with open(
        run_dir / "config.json",
        "w",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    best_val_spearman = -999.0
    best_epoch = None

    history = []

    optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        num_batches = 0

        for step, (tok1, tok2, score) in enumerate(
            train_loader,
            start=1,
        ):
            tok1 = {
                k: v.to(device)
                for k, v in tok1.items()
            }

            tok2 = {
                k: v.to(device)
                for k, v in tok2.items()
            }

            # STSBenchmark gold score: 0..5
            target = (
                score.to(device) / 5.0
            )

            z1, z2 = model(tok1, tok2)

            pred = F.cosine_similarity(
                z1,
                z2,
                dim=-1,
            )

            # Reconstructed STS objective:
            # MSE between cosine similarity and normalized gold score
            loss = F.mse_loss(
                pred,
                target,
            )

            loss = loss / args.grad_accum
            loss.backward()

            if (
                step % args.grad_accum == 0
                or step == len(train_loader)
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += (
                loss.item() * args.grad_accum
            )

            num_batches += 1

            if (
                step % args.log_every == 0
                or step == len(train_loader)
            ):
                print(
                    f"epoch {epoch} "
                    f"step {step}/{len(train_loader)} "
                    f"loss "
                    f"{running_loss / num_batches:.6f}"
                )

        train_loss = running_loss / num_batches

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_spearman":
                val_metrics["spearman"],
            "val_pearson":
                val_metrics["pearson"],
        }

        history.append(row)

        print(
            f"\nEpoch {epoch}: "
            f"loss={train_loss:.6f} | "
            f"val Spearman="
            f"{val_metrics['spearman']:.6f} | "
            f"val Pearson="
            f"{val_metrics['pearson']:.6f}\n"
        )

        if (
            val_metrics["spearman"]
            > best_val_spearman
        ):
            best_val_spearman = (
                val_metrics["spearman"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "epoch":
                        epoch,
                    "val_spearman":
                        best_val_spearman,
                },
                best_path,
            )

            print(
                "Saved new best checkpoint:",
                best_path,
            )

    # -----------------------------------------------------
    # Load best
    # -----------------------------------------------------

    ckpt = torch.load(
        best_path,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    # -----------------------------------------------------
    # Final evaluation
    # -----------------------------------------------------

    val_metrics, _, _ = evaluate(
        model,
        val_loader,
        device,
    )

    test_metrics, preds, labels = evaluate(
        model,
        test_loader,
        device,
    )

    result = {
        "experiment": (
            args.experiment_name
            if args.experiment_name is not None
            else args.pooling
        ),
        "seed": args.seed,
        "pooling": args.pooling,
        "best_epoch": best_epoch,
        "best_val_spearman":
            best_val_spearman,
        "final_val_spearman":
            val_metrics["spearman"],
        "final_val_pearson":
            val_metrics["pearson"],
        "test_spearman":
            test_metrics["spearman"],
        "test_pearson":
            test_metrics["pearson"],
    }

    print("\n================================")
    print("FINAL RESULTS")
    print("================================")

    for k, v in result.items():
        print(f"{k}: {v}")

    with open(
        run_dir / "metrics.json",
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    pd.DataFrame({
        "prediction": preds,
        "true": labels,
    }).to_csv(
        run_dir / "test_predictions.csv",
        index=False,
    )

    pd.DataFrame(history).to_csv(
        run_dir / "history.csv",
        index=False,
    )

    print("\nSaved to:", run_dir)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pooling",
        choices=["mean", "FLaG", "FLaG_Hann", "STFT_FLaG", "STFT_FLaG_Pos"],
        required=True,
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--model_path",
        default=(
            "/home/data/home/wwr_lumos/"
            "models/roberta-base"
        ),
    )

    parser.add_argument(
        "--data_path",
        default=(
            "/home/data/home/wwr_lumos/"
            "AMPCliff/data/text/stsbenchmark"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "/home/data/home/wwr_lumos/"
            "AMPCliff/outputs/text/stsbenchmark"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )
    
    parser.add_argument(
        "--stft_win_length",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--stft_hop_length",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--pool_lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--limit_train",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--limit_val",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--limit_test",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--stft_window_type",
        choices=["rect", "hann"],
        default="rect",
    )

    parser.add_argument(
        "--stft_center",
        action="store_true",
    )

    parser.add_argument(
        "--fixed_fft_length",
        type=int,
        default=None,
        help=(
            "Use a fixed global FFT/iFFT length for FLaG. "
            "None keeps the original batch-dependent FFT length."
        ),
    )

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
