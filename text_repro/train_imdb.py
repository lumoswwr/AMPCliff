import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from AMPCliff.factory.pooling.flag_pooling import (
    FFTLatentAttentionGatePooling,
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

class IMDBDataset(Dataset):
    def __init__(self, split):
        self.data = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return {
            "text": x["text"],
            "label": int(x["label"]),
        }


class IMDBCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        tokens = self.tokenizer(
            [x["text"] for x in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor(
            [x["label"] for x in batch],
            dtype=torch.long,
        )
        return tokens, labels


# ---------------------------------------------------------
# Pooling
# ---------------------------------------------------------

class MeanPooling(nn.Module):
    def forward(self, features, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(features.dtype)
        summed = (features * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom


class IMDBEncoder(nn.Module):
    """
    IMDB-specific RoBERTa encoder.

    Do not reuse SprintEncoder here: Sprint intentionally uses a frozen
    backbone and a different text-task control configuration.

    Paper-aligned FLaG settings:
      - 8 latent queries
      - 4 attention heads
      - dropout = 0.1
      - residual gate
      - masked max time pooling
      - post-pooling LayerNorm for the text domain
      - end-to-end RoBERTa fine-tuning
    """

    def __init__(
        self,
        model_path,
        pooling,
        pool_dropout=0.1,
        mean_anchor_beta_init=0.1,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
        )

        d_model = self.backbone.config.hidden_size

        if pooling == "mean":
            self.pool = MeanPooling()

        elif pooling in {
            "FLaG",
            "FLaG_AlignmentAnchor",
        }:
            self.pool = FFTLatentAttentionGatePooling(
                d_model=d_model,
                num_latents=8,
                num_heads=4,
                dropout=pool_dropout,
                time_pool="max",
                gate_residual=True,
                eps=1e-6,
                use_gate=True,
                use_latent=True,
                # The paper explicitly uses LayerNorm after pooling for
                # image/text domains. Sprint's published frozen protocol does
                # not supply the IMDB configuration and must not be reused.
                post_pool_norm=True,
                window_type=None,
                fixed_fft_length=None,
                remove_dc=False,
                dc_only=False,
                mean_anchor_residual=(
                    pooling == "FLaG_AlignmentAnchor"
                ),
                mean_anchor_beta_init=mean_anchor_beta_init,
                mean_alignment_anchor=(
                    pooling == "FLaG_AlignmentAnchor"
                ),
            )

        else:
            raise ValueError(
                f"Unsupported IMDB pooling: {pooling}"
            )

        self.pooling_name = pooling

    def forward(self, tokens):
        hidden = self.backbone(
            **tokens
        ).last_hidden_state

        return self.pool(
            hidden,
            tokens["attention_mask"],
        )


class IMDBClassifier(nn.Module):
    def __init__(
        self,
        model_path,
        pooling,
        pool_dropout=0.1,
        mean_anchor_beta_init=0.1,
    ):
        super().__init__()

        self.encoder = IMDBEncoder(
            model_path=model_path,
            pooling=pooling,
            pool_dropout=pool_dropout,
            mean_anchor_beta_init=mean_anchor_beta_init,
        )

        hidden_size = (
            self.encoder.backbone.config.hidden_size
        )

        self.classifier = nn.Linear(
            hidden_size,
            2,
        )

    def forward(self, tokens):
        z = self.encoder(tokens)
        return self.classifier(z)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    labels_all = []
    preds_all = []

    for tokens, labels in loader:
        tokens = {
            k: v.to(device)
            for k, v in tokens.items()
        }

        logits = model(tokens)

        preds = (
            logits.argmax(dim=-1)
            .detach()
            .cpu()
            .numpy()
        )

        labels_all.extend(
            labels.numpy().tolist()
        )
        preds_all.extend(
            preds.tolist()
        )

    return {
        "accuracy": float(
            accuracy_score(
                labels_all,
                preds_all,
            )
        ),
        "f1": float(
            f1_score(
                labels_all,
                preds_all,
            )
        ),
    }


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train(args):
    seed_everything(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("IMDB paper-aligned protocol")
    print("=" * 72)
    print("device:", device)
    print("seed:", args.seed)
    print("pooling:", args.pooling)
    print("epochs:", args.epochs)
    print("max_length:", args.max_length)
    print(
        "micro/effective batch:",
        args.batch_size,
        "/",
        args.batch_size * args.grad_accum,
    )
    print("backbone_lr:", args.backbone_lr)
    print("pool_lr:", args.pool_lr)
    print("pool_dropout:", args.pool_dropout)
    print("post_pool_norm: True")
    print()

    raw = load_from_disk(
        args.data_path
    )

    if "train" not in raw or "test" not in raw:
        raise ValueError(
            "IMDB dataset must contain train and test splits."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    full_train = IMDBDataset(
        raw["train"]
    )

    val_size = int(
        round(
            len(full_train)
            * args.val_ratio
        )
    )
    train_size = (
        len(full_train)
        - val_size
    )

    split_generator = (
        torch.Generator()
        .manual_seed(args.seed)
    )

    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=split_generator,
    )

    test_set = IMDBDataset(
        raw["test"]
    )

    print(
        "dataset sizes:",
        f"train={len(train_set)}",
        f"val={len(val_set)}",
        f"test={len(test_set)}",
    )

    collator = IMDBCollator(
        tokenizer,
        args.max_length,
    )

    loader_generator = (
        torch.Generator()
        .manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model = IMDBClassifier(
        model_path=args.model_path,
        pooling=args.pooling,
        pool_dropout=args.pool_dropout,
        mean_anchor_beta_init=(
            args.mean_anchor_beta_init
        ),
    ).to(device)

    # -----------------------------------------------------
    # Differential learning rates
    # -----------------------------------------------------

    parameter_groups = [
        {
            "params": (
                model.encoder.backbone.parameters()
            ),
            "lr": args.backbone_lr,
            "weight_decay": args.weight_decay,
        },
    ]

    pool_params = []
    scalar_params = []

    for name, param in (
        model.encoder.pool.named_parameters()
    ):
        if not param.requires_grad:
            continue

        if name.endswith(
            "mean_anchor_beta_unbounded"
        ):
            scalar_params.append(param)
        else:
            pool_params.append(param)

    head_params = list(
        model.classifier.parameters()
    )

    if pool_params or head_params:
        parameter_groups.append({
            "params": pool_params + head_params,
            "lr": args.pool_lr,
            "weight_decay": args.weight_decay,
        })

    if scalar_params:
        parameter_groups.append({
            "params": scalar_params,
            "lr": args.pool_lr,
            "weight_decay": 0.0,
        })

    optimizer = torch.optim.AdamW(
        parameter_groups,
        eps=1e-8,
    )

    updates_per_epoch = math.ceil(
        len(train_loader)
        / args.grad_accum
    )

    total_steps = (
        updates_per_epoch
        * args.epochs
    )

    warmup_steps = int(
        total_steps
        * args.warmup_ratio
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    )

    loss_fn = nn.CrossEntropyLoss()

    run_dir = (
        Path(args.output_dir)
        / args.experiment_name
        / f"seed_{args.seed}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        run_dir / "config.json",
        "w",
    ) as f:
        json.dump(
            vars(args),
            f,
            indent=2,
        )

    best_val_acc = -1.0
    best_val_f1 = -1.0
    best_epoch = None
    best_path = (
        run_dir
        / "best_model.pt"
    )

    history = []

    optimizer.zero_grad()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        running_loss = 0.0
        num_batches = 0

        bar = tqdm(
            train_loader,
            desc=(
                f"{args.pooling} "
                f"epoch {epoch}/{args.epochs}"
            ),
        )

        for step, (
            tokens,
            labels,
        ) in enumerate(
            bar,
            start=1,
        ):
            tokens = {
                k: v.to(device)
                for k, v in tokens.items()
            }
            labels = labels.to(device)

            logits = model(tokens)

            raw_loss = loss_fn(
                logits,
                labels,
            )

            loss = (
                raw_loss
                / args.grad_accum
            )

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
                raw_loss.item()
            )
            num_batches += 1

            bar.set_postfix(
                loss=(
                    f"{running_loss / num_batches:.4f}"
                )
            )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        train_loss = (
            running_loss
            / max(num_batches, 1)
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": (
                val_metrics["accuracy"]
            ),
            "val_f1": (
                val_metrics["f1"]
            ),
        }

        if hasattr(
            model.encoder.pool,
            "mean_anchor_beta_unbounded",
        ):
            row[
                "mean_anchor_beta"
            ] = float(
                model.encoder.pool
                .mean_anchor_beta_unbounded
                .detach()
                .cpu()
            )

        history.append(row)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        # The paper uses validation accuracy for IMDB checkpoint
        # selection. Keep the first checkpoint on an exact tie.
        if (
            val_metrics["accuracy"]
            > best_val_acc
        ):
            best_val_acc = (
                val_metrics["accuracy"]
            )
            best_val_f1 = (
                val_metrics["f1"]
            )
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                best_path,
            )

    model.load_state_dict(
        torch.load(
            best_path,
            map_location=device,
        )
    )

    test_metrics = evaluate(
        model,
        test_loader,
        device,
    )

    result = {
        "protocol": (
            "IMDB end-to-end RoBERTa-base; "
            "90:10 train/val; paper-aligned FLaG"
        ),
        "pooling": args.pooling,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "val_accuracy": best_val_acc,
        "val_f1": best_val_f1,
        "test_accuracy": (
            test_metrics["accuracy"]
        ),
        "test_f1": (
            test_metrics["f1"]
        ),
        "epochs": args.epochs,
        "max_length": args.max_length,
        "micro_batch_size": (
            args.batch_size
        ),
        "grad_accum": args.grad_accum,
        "effective_batch_size": (
            args.batch_size
            * args.grad_accum
        ),
        "backbone_lr": (
            args.backbone_lr
        ),
        "pool_lr": args.pool_lr,
        "pool_dropout": (
            args.pool_dropout
        ),
        "post_pool_norm": True,
    }

    if hasattr(
        model.encoder.pool,
        "mean_anchor_beta_unbounded",
    ):
        result[
            "mean_anchor_beta"
        ] = float(
            model.encoder.pool
            .mean_anchor_beta_unbounded
            .detach()
            .cpu()
        )

    with open(
        run_dir / "metrics.json",
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    with open(
        run_dir / "history.csv",
        "w",
        newline="",
    ) as f:
        fieldnames = list(
            history[0].keys()
        )
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(history)

    print()
    print("=" * 72)
    print("FINAL")
    print(json.dumps(
        result,
        indent=2,
    ))
    print("saved to:", run_dir)
    print("=" * 72)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        default="data/text/imdb",
    )
    parser.add_argument(
        "--model_path",
        default=(
            "/home/data/home/wwr_lumos/"
            "models/roberta-base"
        ),
    )
    parser.add_argument(
        "--pooling",
        choices=[
            "mean",
            "FLaG",
            "FLaG_AlignmentAnchor",
        ],
        required=True,
    )
    parser.add_argument(
        "--experiment_name",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "outputs/text/imdb/"
            "paper_protocol"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    # Main-paper IMDB protocol: end-to-end RoBERTa for 3 epochs.
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    # IMDB reviews are long. Use RoBERTa's full 512-token context instead
    # of the earlier 256-token reproduction shortcut.
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
    )

    # Keep an effective batch of 32 while fitting full-context RoBERTa-base
    # comfortably on a 16 GB V100.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
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
        "--pool_dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--mean_anchor_beta_init",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
