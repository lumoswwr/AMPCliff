import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel

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

class SprintDataset(Dataset):
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
            "label": float(x["label"]),
        }


class SprintCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        s1 = [x["sentence1"] for x in batch]
        s2 = [x["sentence2"] for x in batch]

        labels = torch.tensor(
            [x["label"] for x in batch],
            dtype=torch.float32,
        )

        # Tokenize both sides together so dynamic padding uses
        # the same tensor length for both sides of every pair.
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

        return tok1, tok2, labels


# ---------------------------------------------------------
# Pooling
# ---------------------------------------------------------

class MeanPooling(nn.Module):
    def forward(self, features, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(features.dtype)

        summed = (features * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)

        return summed / denom


# ---------------------------------------------------------
# Frozen RoBERTa + pooling
# ---------------------------------------------------------

class SprintEncoder(nn.Module):
    def __init__(
        self,
        model_path,
        pooling,
        stft_win_length=16,
        stft_hop_length=16,
        stft_window_type="rect",
        stft_center=False,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        d_model = self.backbone.config.hidden_size

        if pooling == "mean":
            self.pool = MeanPooling()

        elif pooling == "FLaG":
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
                window_type=None,
                fixed_fft_length=None,
            )

        elif pooling == "STFT_FLaG":
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
                use_frame_positional_encoding=False,
                max_frame_positions=32,
                local_window_type=stft_window_type,
                center_frames=stft_center,
            )

        else:
            raise ValueError(
                f"Unknown pooling: {pooling}"
            )

        self.pooling_name = pooling

    def train(self, mode=True):
        super().train(mode)

        # The paper specifies a frozen RoBERTa backbone for Sprint.
        # Keep it in eval mode as a deterministic frozen feature extractor.
        self.backbone.eval()

        return self

    def forward(self, tokens1, tokens2):
        combined = {}

        for key in tokens1:
            if key in tokens2:
                combined[key] = torch.cat(
                    [tokens1[key], tokens2[key]],
                    dim=0,
                )

        self.backbone.eval()

        with torch.no_grad():
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


class CosineLogitHead(nn.Module):
    """
    Monotone cosine-logit scorer.

    Duplicate pairs must never receive lower logits merely because a free
    affine weight changes sign. The public supplement names a cosine-logit
    scorer but does not release the Sprint implementation, so we use the
    minimal monotone parameterization

        logit = positive_scale * cosine + bias

    where positive_scale = softplus(raw_scale).

    This preserves cosine ordering while still allowing BCE training to learn
    calibration. The exact scorer parameterization is therefore recorded as a
    reproduction choice rather than an author-reported implementation detail.
    """

    def __init__(self, scale_init=1.0, bias_init=0.0):
        super().__init__()

        if scale_init <= 0:
            raise ValueError("scale_init must be positive.")

        # Inverse softplus so softplus(raw_scale) starts at scale_init.
        raw_scale_init = np.log(
            np.expm1(scale_init)
        )

        self.raw_scale = nn.Parameter(
            torch.tensor(
                float(raw_scale_init),
                dtype=torch.float32,
            )
        )

        self.bias = nn.Parameter(
            torch.tensor(
                float(bias_init),
                dtype=torch.float32,
            )
        )

    def positive_scale(self):
        return F.softplus(
            self.raw_scale
        )

    def forward(self, z1, z2):
        cosine = F.cosine_similarity(
            z1,
            z2,
            dim=-1,
        )

        logits = (
            self.positive_scale()
            * cosine
            + self.bias
        )

        return logits, cosine


class SprintPairClassifier(nn.Module):
    def __init__(
        self,
        model_path,
        pooling,
        stft_win_length=16,
        stft_hop_length=16,
        stft_window_type="rect",
        stft_center=False,
    ):
        super().__init__()

        self.encoder = SprintEncoder(
            model_path=model_path,
            pooling=pooling,
            stft_win_length=stft_win_length,
            stft_hop_length=stft_hop_length,
            stft_window_type=stft_window_type,
            stft_center=stft_center,
        )

        self.head = CosineLogitHead()

    @property
    def backbone(self):
        return self.encoder.backbone

    @property
    def pool(self):
        return self.encoder.pool

    def forward(self, tokens1, tokens2):
        z1, z2 = self.encoder(
            tokens1,
            tokens2,
        )

        return self.head(
            z1,
            z2,
        )


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def classification_metrics(logits, labels, cosines=None):
    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probs = 1.0 / (
        1.0 + np.exp(
            -np.clip(
                logits,
                -50.0,
                50.0,
            )
        )
    )

    # BCEWithLogits uses zero logit as the natural 0.5 threshold.
    preds = (
        logits >= 0.0
    ).astype(np.int64)

    return {
        "accuracy": float(
            accuracy_score(
                labels,
                preds,
            )
        ),
        "average_precision": float(
            average_precision_score(
                labels,
                probs,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                preds,
                zero_division=0,
            )
        ),
        # Diagnostics only; not used for checkpoint selection.
        "precision": float(
            precision_score(
                labels,
                preds,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                preds,
                zero_division=0,
            )
        ),
        "predicted_positive_rate": float(
            preds.mean()
        ),
        "cosine_average_precision": (
            float(
                average_precision_score(
                    labels,
                    np.asarray(
                        cosines,
                        dtype=np.float64,
                    ),
                )
            )
            if cosines is not None
            else None
        ),
    }


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_logits = []
    all_labels = []
    all_cosines = []

    for tok1, tok2, labels in loader:
        tok1 = {
            k: v.to(device)
            for k, v in tok1.items()
        }

        tok2 = {
            k: v.to(device)
            for k, v in tok2.items()
        }

        logits, cosine = model(
            tok1,
            tok2,
        )

        all_logits.extend(
            logits.detach().cpu().numpy().tolist()
        )

        all_cosines.extend(
            cosine.detach().cpu().numpy().tolist()
        )

        all_labels.extend(
            labels.numpy().tolist()
        )

    metrics = classification_metrics(
        all_logits,
        all_labels,
        cosines=all_cosines,
    )

    return (
        metrics,
        all_logits,
        all_labels,
        all_cosines,
    )


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train(args):
    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("device:", device)
    print("seed:", args.seed)
    print("pooling:", args.pooling)

    print("\nSprint protocol:")
    print("- frozen RoBERTa-base")
    print("- train pooling + monotone cosine-logit head")
    print("- BCEWithLogitsLoss")
    print("- checkpoint criterion: validation accuracy")
    print("- official test evaluated only after checkpoint selection")

    if args.pooling == "STFT_FLaG":
        print("\nFrozen E12:")
        print("- win_length:", args.stft_win_length)
        print("- hop_length:", args.stft_hop_length)
        print("- window:", args.stft_window_type)
        print("- center:", args.stft_center)

        if (
            args.stft_win_length != 16
            or args.stft_hop_length != 16
            or args.stft_window_type != "rect"
            or args.stft_center
        ):
            raise ValueError(
                "Sprint STFT_FLaG is frozen to E12: "
                "win=16, hop=16, rect, center=False."
            )

    raw = load_from_disk(
        args.data_path
    )

    required_splits = {
        "train",
        "validation",
        "test",
    }

    missing = (
        required_splits
        - set(raw.keys())
    )

    if missing:
        raise ValueError(
            f"Missing dataset splits: {sorted(missing)}"
        )

    print("\nDataset sizes:")
    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        print(
            split_name,
            len(raw[split_name]),
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    train_ds = SprintDataset(
        raw["train"],
        args.limit_train,
    )

    val_ds = SprintDataset(
        raw["validation"],
        args.limit_val,
    )

    test_ds = SprintDataset(
        raw["test"],
        args.limit_test,
    )

    collator = SprintCollator(
        tokenizer,
        max_length=args.max_length,
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model = SprintPairClassifier(
        model_path=args.model_path,
        pooling=args.pooling,
        stft_win_length=args.stft_win_length,
        stft_hop_length=args.stft_hop_length,
        stft_window_type=args.stft_window_type,
        stft_center=args.stft_center,
    ).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    backbone_trainable = sum(
        p.numel()
        for p in model.backbone.parameters()
        if p.requires_grad
    )

    pool_trainable = sum(
        p.numel()
        for p in model.pool.parameters()
        if p.requires_grad
    )

    head_trainable = sum(
        p.numel()
        for p in model.head.parameters()
        if p.requires_grad
    )

    print("\nParameter sanity:")
    print("total params:", total_params)
    print("trainable params:", trainable_params)
    print("backbone trainable:", backbone_trainable)
    print("pool trainable:", pool_trainable)
    print("head trainable:", head_trainable)
    print(
        "head positive scale:",
        float(
            model.head.positive_scale()
            .detach()
            .cpu()
        ),
    )
    print(
        "head bias:",
        float(
            model.head.bias
            .detach()
            .cpu()
        ),
    )

    if backbone_trainable != 0:
        raise RuntimeError(
            "RoBERTa backbone must be frozen for Sprint."
        )

    trainable = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

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

    best_path = (
        run_dir
        / "best_model.pt"
    )

    config = vars(args).copy()

    config["protocol_notes"] = {
        "paper_specified": {
            "backbone": "RoBERTa-base bi-encoder",
            "backbone_frozen": True,
            "trainable": "pooling module + lightweight head",
            "epochs": 10,
            "learning_rate": 1e-3,
            "loss": "BCEWithLogitsLoss",
            "checkpoint_criterion": "validation accuracy",
            "metrics": [
                "accuracy",
                "average_precision",
                "f1",
            ],
            "max_length": 128,
        },
        "reproduction_choices_not_reported_in_public_release": {
            "split": (
                "fixed stratified 90/10 split of official validation; "
                "random_state=42; shared across model seeds"
            ),
            "cosine_logit_head": (
                "monotone affine cosine scorer: "
                "softplus(raw_scale) * cosine + bias; scale_init=1.0"
            ),
            "optimizer": "AdamW",
            "weight_decay": args.weight_decay,
            "scheduler": "none",
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "frozen_backbone_mode": "eval + no_grad",
            "decision_threshold": "logit >= 0",
            "checkpoint_tie_break": "keep earliest epoch",
        },
        "stft_frozen_e12": {
            "win_length": 16,
            "hop_length": 16,
            "window_type": "rect",
            "center": False,
            "frame_positional_encoding": False,
            "num_latents": 8,
            "num_heads": 4,
        },
    }

    with open(
        run_dir / "config.json",
        "w",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    criterion = nn.BCEWithLogitsLoss()

    best_val_accuracy = -1.0
    best_epoch = None

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        running_loss = 0.0
        num_batches = 0

        for step, (
            tok1,
            tok2,
            labels,
        ) in enumerate(
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

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            logits, _ = model(
                tok1,
                tok2,
            )

            # Deliberately unweighted. The public Sprint protocol
            # reports BCEWithLogits and does not report pos_weight.
            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable,
                max_norm=1.0,
            )

            optimizer.step()

            running_loss += loss.item()
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

        train_loss = (
            running_loss
            / max(num_batches, 1)
        )

        (
            val_metrics,
            _,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{
                f"val_{k}": v
                for k, v in val_metrics.items()
            },
        }

        history.append(
            row
        )

        print(
            f"\nEpoch {epoch}: "
            f"loss={train_loss:.6f} | "
            f"val Acc={val_metrics['accuracy']:.6f} | "
            f"val AP={val_metrics['average_precision']:.6f} | "
            f"val F1={val_metrics['f1']:.6f} | "
            f"cosAP={val_metrics['cosine_average_precision']:.6f} | "
            f"pred+={val_metrics['predicted_positive_rate']:.6f} | "
            f"scale={float(model.head.positive_scale().detach().cpu()):.6f} | "
            f"bias={float(model.head.bias.detach().cpu()):.6f}\n"
        )

        # Strictly greater keeps the earliest checkpoint on ties.
        # No secondary metric is used because the paper specifies
        # validation accuracy as the selection criterion.
        if (
            val_metrics["accuracy"]
            > best_val_accuracy
        ):
            best_val_accuracy = (
                val_metrics["accuracy"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "epoch":
                        epoch,
                    "val_accuracy":
                        best_val_accuracy,
                },
                best_path,
            )

            print(
                "Saved new best checkpoint:",
                best_path,
            )

    checkpoint = torch.load(
        best_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    (
        val_metrics,
        _,
        _,
        _,
    ) = evaluate(
        model,
        val_loader,
        device,
    )

    test_metrics = None
    test_logits = None
    test_labels = None
    test_cosines = None

    if not args.skip_test:
        (
            test_metrics,
            test_logits,
            test_labels,
            test_cosines,
        ) = evaluate(
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
        "best_val_accuracy": best_val_accuracy,
        **{
            f"final_val_{k}": v
            for k, v in val_metrics.items()
        },
        **(
            {
                f"test_{k}": v
                for k, v in test_metrics.items()
            }
            if test_metrics is not None
            else {}
        ),
    }

    print("\n================================")
    print("FINAL RESULTS")
    print("================================")

    for key, value in result.items():
        print(
            f"{key}: {value}"
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

    if not args.skip_test:
        probs = 1.0 / (
            1.0 + np.exp(
                -np.clip(
                    np.asarray(
                        test_logits,
                        dtype=np.float64,
                    ),
                    -50.0,
                    50.0,
                )
            )
        )

        preds = (
            np.asarray(
                test_logits
            )
            >= 0.0
        ).astype(np.int64)

        pd.DataFrame({
            "logit": test_logits,
            "probability": probs,
            "prediction": preds,
            "true": np.asarray(
                test_labels,
                dtype=np.int64,
            ),
            "cosine": test_cosines,
        }).to_csv(
            run_dir
            / "test_predictions.csv",
            index=False,
        )

    pd.DataFrame(
        history
    ).to_csv(
        run_dir
        / "history.csv",
        index=False,
    )

    print(
        "\nSaved to:",
        run_dir,
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pooling",
        choices=[
            "mean",
            "FLaG",
            "STFT_FLaG",
        ],
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
            "/home/data/home/wwr_lumos/AMPCliff/"
            "data/text/sprintduplicatequestions_adaptation"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "/home/data/home/wwr_lumos/AMPCliff/"
            "outputs/text/sprintduplicatequestions"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    # The Sprint batch size is not reported in the public supplement.
    # 32 is a documented reproduction choice, not an author-reported value.
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
    )

    # Not reported for Sprint. Keep explicit and recorded.
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=200,
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
        "--skip_test",
        action="store_true",
        help=(
            "Do not evaluate the official test split. "
            "Use during protocol debugging and model selection."
        ),
    )

    # Frozen E12 values. These remain CLI-visible for config logging
    # but STFT_FLaG refuses non-E12 values in train().
    parser.add_argument(
        "--stft_win_length",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--stft_hop_length",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--stft_window_type",
        choices=[
            "rect",
            "hann",
        ],
        default="rect",
    )

    parser.add_argument(
        "--stft_center",
        action="store_true",
    )

    args = parser.parse_args()

    train(
        args
    )


if __name__ == "__main__":
    main()
