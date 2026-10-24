#!/usr/bin/env python3
"""Read-only STSB checkpoint check: does evaluation batch padding change ranking?

For each already-trained Mean/FLaG-Zero/FLaG pure-LTR seed-0 checkpoint,
evaluate EXACTLY the same first N validation sentence pairs using DataLoader
batch_size 1 and batch_size N. Large changes could indicate batch-dependent
frequency processing due to dynamic FFT length and padding.

No training, no gradient, no test, no new result file, no model modifications.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from train_author_text_protocol import PairCollator, PairDataset, seed_everything
from run_stsb_ltr_v2 import evaluate, make_model


def load_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path,
                   default=Path("outputs/text/stsb_pure_softsp_cosine_v1"))
    p.add_argument("--model_path",
                   default="/home/data/home/wwr_lumos/models/roberta-base")
    p.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.n < 4:
        p.error("--n >= 4 required")

    seed_everything(args.seed)
    raw = load_from_disk(args.stsb_data_path)
    ds = PairDataset(raw["validation"], "stsb")
    n = min(args.n, len(ds))
    ds = Subset(ds, list(range(n)))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    collate = PairCollator(tokenizer, 128, torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] same {n} validation examples, batch sizes 1 vs {n}; "
          f"seed={args.seed}; no test or train.", flush=True)

    for model_name in ("mean", "flag_zero", "flag"):
        checkpoint = (
            args.root / model_name / "pure_softsp_cosine" /
            f"seed_{args.seed}" / "best_model.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = make_model(model_name, args, device)
        model.load_state_dict(load_state(checkpoint), strict=True)

        outputs = []
        for bs in (1, n):
            loader = DataLoader(
                ds, batch_size=bs, shuffle=False, collate_fn=collate,
                num_workers=0,
            )
            metrics, arrays = evaluate(model, loader, device)
            outputs.append((metrics, arrays))
        (small, a), (large, b) = outputs
        if not np.array_equal(a["labels"], b["labels"]):
            raise AssertionError(f"Label order mismatch for {model_name}")
        diffs = np.abs(a["cosine"] - b["cosine"])
        sp_delta = large["spearman"] - small["spearman"]
        print(
            f"[{model_name:9}] cosine_max_abs_diff={diffs.max():.8f} "
            f"cosine_mean_abs_diff={diffs.mean():.8f} "
            f"spearman_batch1={small['spearman']:.6f} "
            f"spearman_batchN={large['spearman']:.6f} "
            f"spearman_change={sp_delta:+.6f}",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[complete] padding invariance smoke-check (no updates).", flush=True)


if __name__ == "__main__":
    main()
