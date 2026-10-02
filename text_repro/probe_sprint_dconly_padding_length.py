#!/usr/bin/env python3
"""
Audit whether the very strong Sprint DC-only result is driven by
dynamic-padding / sentence-length shortcuts.

Checks:
1) checkpoint/config sanity: frozen RoBERTa + FLaG + exact dc_only
2) same checkpoint, validation AP under several dynamic-padding batch sizes
3) same batch size but different sample grouping/order
4) static padding to max_length=128
5) official test AP with dynamic vs static padding
6) a length-only logistic-regression baseline

Interpretation:
- If DC-only AP is stable across batch size/grouping/static padding, the score
  is not mainly caused by batch-dependent FFT length.
- If a length-only classifier is weak while DC-only FLaG is strong, raw token
  length alone cannot explain the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from datasets import load_from_disk
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = _REPO_ROOT.parent
for _p in (_REPO_PARENT, _REPO_ROOT / "text_repro"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from train_sprint import (
    SprintDataset,
    SprintCollator,
    SprintPairClassifier,
    evaluate,
)


class StaticSprintCollator:
    """Same pair collation, but always pad every sentence to max_length."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch):
        s1 = [x["sentence1"] for x in batch]
        s2 = [x["sentence2"] for x in batch]
        labels = torch.tensor(
            [x["label"] for x in batch],
            dtype=torch.float32,
        )

        tokens = self.tokenizer(
            s1 + s2,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        b = len(batch)
        tok1 = {k: v[:b] for k, v in tokens.items()}
        tok2 = {k: v[b:] for k, v in tokens.items()}
        return tok1, tok2, labels


def load_dc_only_model(
    model_path: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    model = SprintPairClassifier(
        model_path=str(model_path),
        pooling="FLaG",
        finetune_backbone=False,
        dc_only=True,
    ).to(device)

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
    )
    state = ckpt["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def make_loader(
    dataset,
    collator,
    batch_size,
):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )


def eval_ap(model, loader, device):
    metrics, logits, labels, cosines = evaluate(
        model,
        loader,
        device,
    )
    return {
        "ap": float(metrics["average_precision"]),
        "cosine_ap": float(metrics["cosine_average_precision"]),
        "n": int(len(labels)),
        "logits": np.asarray(logits, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
        "cosines": np.asarray(cosines, dtype=np.float64),
    }


def tokenize_lengths(tokenizer, texts, max_length, chunk=2048):
    lengths = []
    for start in range(0, len(texts), chunk):
        part = texts[start:start + chunk]
        enc = tokenizer(
            part,
            padding=False,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_length=True,
        )
        lengths.extend(enc["length"])
    return np.asarray(lengths, dtype=np.float64)


def length_features(tokenizer, split, max_length):
    s1 = list(split["sentence1"])
    s2 = list(split["sentence2"])
    y = np.asarray(split["label"], dtype=np.int64)

    l1 = tokenize_lengths(
        tokenizer,
        s1,
        max_length=max_length,
    )
    l2 = tokenize_lengths(
        tokenizer,
        s2,
        max_length=max_length,
    )

    mn = np.minimum(l1, l2)
    mx = np.maximum(l1, l2)
    diff = np.abs(l1 - l2)
    ratio = mn / np.maximum(mx, 1.0)
    total = l1 + l2

    x = np.column_stack([
        l1,
        l2,
        mn,
        mx,
        diff,
        ratio,
        total,
    ])

    return x, y


def fit_length_baseline(tokenizer, raw, max_length):
    print("\n[length-only baseline] extracting token lengths...")
    x_train, y_train = length_features(
        tokenizer,
        raw["train"],
        max_length,
    )
    x_val, y_val = length_features(
        tokenizer,
        raw["validation"],
        max_length,
    )
    x_test, y_test = length_features(
        tokenizer,
        raw["test"],
        max_length,
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        ),
    )
    clf.fit(x_train, y_train)

    val_score = clf.decision_function(x_val)
    test_score = clf.decision_function(x_test)

    val_ap = float(
        average_precision_score(
            y_val,
            val_score,
        )
    )
    test_ap = float(
        average_precision_score(
            y_test,
            test_score,
        )
    )

    # Very simple no-training reference: similar lengths -> larger score.
    val_similarity = -np.abs(
        x_val[:, 0] - x_val[:, 1]
    )
    test_similarity = -np.abs(
        x_test[:, 0] - x_test[:, 1]
    )

    return {
        "logreg_val_ap": val_ap,
        "logreg_test_ap": test_ap,
        "negative_abs_length_diff_val_ap": float(
            average_precision_score(
                y_val,
                val_similarity,
            )
        ),
        "negative_abs_length_diff_test_ap": float(
            average_precision_score(
                y_test,
                test_similarity,
            )
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/text/sprintduplicatequestions/experiments/"
            "sprint_frozen_flag_dconly/seed_0/best_model.pt"
        ),
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(
            "outputs/text/sprintduplicatequestions/experiments/"
            "sprint_frozen_flag_dconly/seed_0/config.json"
        ),
    )
    ap.add_argument(
        "--model_path",
        type=Path,
        default=Path(
            "/home/data/home/wwr_lumos/models/roberta-base"
        ),
    )
    ap.add_argument(
        "--data_path",
        type=Path,
        default=Path(
            "/home/data/home/wwr_lumos/AMPCliff/"
            "data/text/sprintduplicatequestions_adaptation"
        ),
    )
    ap.add_argument(
        "--max_length",
        type=int,
        default=128,
    )
    ap.add_argument(
        "--dynamic_batch_sizes",
        type=int,
        nargs="+",
        default=[1, 8, 32, 64, 128],
    )
    ap.add_argument(
        "--grouping_batch_size",
        type=int,
        default=64,
    )
    ap.add_argument(
        "--skip_length_baseline",
        action="store_true",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/text/dc_training_ablation/"
            "sprint_dconly_padding_length_audit_seed0.json"
        ),
    )
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)

    cfg = json.loads(
        args.config.read_text(encoding="utf-8")
    )

    print("============================================================")
    print("Sprint DC-only padding / length audit")
    print("============================================================")
    print("checkpoint:", args.checkpoint)
    print("finetune_backbone:", cfg.get("finetune_backbone"))
    print("remove_dc:", cfg.get("remove_dc"))
    print("dc_only:", cfg.get("dc_only"))

    if cfg.get("finetune_backbone", False):
        raise RuntimeError(
            "Expected frozen backbone, but finetune_backbone=true."
        )
    if cfg.get("remove_dc", False):
        raise RuntimeError(
            "Expected DC-only, but remove_dc=true."
        )
    if not cfg.get("dc_only", False):
        raise RuntimeError(
            "Expected dc_only=true in checkpoint config."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        local_files_only=True,
    )
    raw = load_from_disk(str(args.data_path))

    val_ds = SprintDataset(raw["validation"])
    test_ds = SprintDataset(raw["test"])

    dynamic_collator = SprintCollator(
        tokenizer,
        max_length=args.max_length,
    )
    static_collator = StaticSprintCollator(
        tokenizer,
        max_length=args.max_length,
    )

    model = load_dc_only_model(
        args.model_path,
        args.checkpoint,
        device,
    )

    results = {
        "config_audit": {
            "finetune_backbone": bool(
                cfg.get("finetune_backbone", False)
            ),
            "remove_dc": bool(
                cfg.get("remove_dc", False)
            ),
            "dc_only": bool(
                cfg.get("dc_only", False)
            ),
        },
        "validation_dynamic_batch_size": {},
    }

    print("\n[1] Validation AP across dynamic-padding batch sizes")
    for bs in args.dynamic_batch_sizes:
        out = eval_ap(
            model,
            make_loader(
                val_ds,
                dynamic_collator,
                bs,
            ),
            device,
        )
        results["validation_dynamic_batch_size"][str(bs)] = {
            "ap": out["ap"],
            "cosine_ap": out["cosine_ap"],
        }
        print(
            f"batch={bs:>3d} "
            f"AP={out['ap']:.6f} "
            f"cosAP={out['cosine_ap']:.6f}"
        )

    print("\n[2] Same batch size, different grouping/order")
    bs = args.grouping_batch_size

    natural = eval_ap(
        model,
        make_loader(
            val_ds,
            dynamic_collator,
            bs,
        ),
        device,
    )

    rng = np.random.RandomState(12345)
    perm = rng.permutation(len(val_ds)).tolist()
    permuted_ds = Subset(
        val_ds,
        perm,
    )
    permuted = eval_ap(
        model,
        make_loader(
            permuted_ds,
            dynamic_collator,
            bs,
        ),
        device,
    )

    results["validation_grouping"] = {
        "batch_size": bs,
        "natural_ap": natural["ap"],
        "permuted_ap": permuted["ap"],
        "absolute_difference": abs(
            natural["ap"] - permuted["ap"]
        ),
    }

    print(
        f"natural order  AP={natural['ap']:.6f}"
    )
    print(
        f"permuted order AP={permuted['ap']:.6f}"
    )
    print(
        "absolute diff  ="
        f"{results['validation_grouping']['absolute_difference']:.6f}"
    )

    print("\n[3] Dynamic padding vs static padding=128")
    val_static = eval_ap(
        model,
        make_loader(
            val_ds,
            static_collator,
            bs,
        ),
        device,
    )

    test_dynamic = eval_ap(
        model,
        make_loader(
            test_ds,
            dynamic_collator,
            bs,
        ),
        device,
    )
    test_static = eval_ap(
        model,
        make_loader(
            test_ds,
            static_collator,
            bs,
        ),
        device,
    )

    results["static_padding"] = {
        "batch_size": bs,
        "validation_dynamic_ap": natural["ap"],
        "validation_static128_ap": val_static["ap"],
        "validation_abs_diff": abs(
            natural["ap"] - val_static["ap"]
        ),
        "test_dynamic_ap": test_dynamic["ap"],
        "test_static128_ap": test_static["ap"],
        "test_abs_diff": abs(
            test_dynamic["ap"] - test_static["ap"]
        ),
    }

    print(
        f"validation dynamic AP = {natural['ap']:.6f}"
    )
    print(
        f"validation static  AP = {val_static['ap']:.6f}"
    )
    print(
        f"test dynamic AP       = {test_dynamic['ap']:.6f}"
    )
    print(
        f"test static AP        = {test_static['ap']:.6f}"
    )

    if not args.skip_length_baseline:
        print("\n[4] Length-only baselines")
        length_results = fit_length_baseline(
            tokenizer,
            raw,
            args.max_length,
        )
        results["length_only"] = length_results
        for k, v in length_results.items():
            print(f"{k}: {v:.6f}")
    else:
        print("\n[4] Length-only baseline skipped")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
