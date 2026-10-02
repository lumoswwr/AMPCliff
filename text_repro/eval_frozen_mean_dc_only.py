#!/usr/bin/env python3
"""
Evaluate the exact DC-only / Mean-pooling control on frozen RoBERTa.

For a token sequence H with valid length T:
    DC = sum_t H_t
    Mean(H) = DC / T

Cosine similarity is invariant to this positive scalar, so masked mean pooling
is the exact DC-only representation for the pairwise cosine metrics used here.

There is intentionally no training for this control:
- frozen RoBERTa + Mean has no trainable pooling parameters on STSB;
- on Sprint, a monotone cosine-logit calibration head cannot change AP ranking,
  so cosine AP is the clean DC-only quantity to compare with FLaG AP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Support direct execution without relying on a pre-set PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = _REPO_ROOT.parent
for _p in (_REPO_PARENT, _REPO_ROOT / "text_repro"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_sts import (
    STSCollator,
    STSDataset,
    SentenceEncoder,
    correlation_metrics,
)
from train_sprint import (
    SprintCollator,
    SprintDataset,
    SprintEncoder,
)


@torch.no_grad()
def eval_stsb(model, loader, device):
    model.eval()
    preds = []
    labels = []

    for tok1, tok2, score in loader:
        tok1 = {k: v.to(device) for k, v in tok1.items()}
        tok2 = {k: v.to(device) for k, v in tok2.items()}

        z1, z2 = model(tok1, tok2)
        cosine = F.cosine_similarity(z1, z2, dim=-1)

        preds.extend(cosine.cpu().numpy().tolist())
        labels.extend(score.numpy().tolist())

    return correlation_metrics(preds, labels)


@torch.no_grad()
def eval_sprint(model, loader, device):
    model.eval()
    cosines = []
    labels = []

    for tok1, tok2, y in loader:
        tok1 = {k: v.to(device) for k, v in tok1.items()}
        tok2 = {k: v.to(device) for k, v in tok2.items()}

        z1, z2 = model(tok1, tok2)
        cosine = F.cosine_similarity(z1, z2, dim=-1)

        cosines.extend(cosine.cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())

    return {
        "cosine_average_precision": float(
            average_precision_score(
                np.asarray(labels, dtype=np.int64),
                np.asarray(cosines, dtype=np.float64),
            )
        )
    }


def make_loader(dataset_cls, collator, split, batch_size):
    return DataLoader(
        dataset_cls(split),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="/home/data/home/wwr_lumos/models/roberta-base",
    )
    parser.add_argument(
        "--stsb_data_path",
        default=(
            "/home/data/home/wwr_lumos/AMPCliff/"
            "data/text/stsbenchmark"
        ),
    )
    parser.add_argument(
        "--sprint_data_path",
        default=(
            "/home/data/home/wwr_lumos/AMPCliff/"
            "data/text/sprintduplicatequestions_adaptation"
        ),
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/text/dc_training_ablation/"
            "mean_dc_only_metrics.json"
        ),
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)
    print("control: frozen RoBERTa + exact DC-only (masked mean)")
    print("training: none; this control is parameter-free for cosine ranking")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    # STSB
    stsb_raw = load_from_disk(args.stsb_data_path)
    stsb_collator = STSCollator(
        tokenizer,
        max_length=args.max_length,
    )
    stsb_model = SentenceEncoder(
        args.model_path,
        "mean",
        freeze_backbone=True,
    ).to(device)

    stsb_val = eval_stsb(
        stsb_model,
        make_loader(
            STSDataset,
            stsb_collator,
            stsb_raw["validation"],
            args.batch_size,
        ),
        device,
    )
    stsb_test = eval_stsb(
        stsb_model,
        make_loader(
            STSDataset,
            stsb_collator,
            stsb_raw["test"],
            args.batch_size,
        ),
        device,
    )

    del stsb_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Sprint
    sprint_raw = load_from_disk(args.sprint_data_path)
    sprint_collator = SprintCollator(
        tokenizer,
        max_length=args.max_length,
    )
    sprint_model = SprintEncoder(
        args.model_path,
        "mean",
        finetune_backbone=False,
    ).to(device)

    sprint_val = eval_sprint(
        sprint_model,
        make_loader(
            SprintDataset,
            sprint_collator,
            sprint_raw["validation"],
            args.batch_size,
        ),
        device,
    )
    sprint_test = eval_sprint(
        sprint_model,
        make_loader(
            SprintDataset,
            sprint_collator,
            sprint_raw["test"],
            args.batch_size,
        ),
        device,
    )

    result = {
        "control": "frozen_roberta_exact_dc_only_mean_pooling",
        "equivalence": (
            "masked mean = sequence DC divided by valid length; "
            "cosine ranking is scale-invariant"
        ),
        "trainable_pooling_parameters": 0,
        "stsb": {
            "validation": stsb_val,
            "test": stsb_test,
        },
        "sprint": {
            "validation": sprint_val,
            "test": sprint_test,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nDC-only / Mean results")
    print(json.dumps(result, indent=2))
    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
