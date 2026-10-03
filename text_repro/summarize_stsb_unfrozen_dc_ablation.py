#!/usr/bin/env python3
"""Summarize unfrozen STS-B Full / No-DC / DC-only / Mean controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_metrics(name: str, seed: int):
    path = Path(
        "outputs/text/stsbenchmark/experiments"
    ) / name / f"seed_{seed}" / "metrics.json"

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open() as f:
        return json.load(f), path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    conditions = [
        ("Full FLaG", "stsb_unfrozen_flag_full"),
        ("No-DC FLaG", "stsb_unfrozen_flag_nodc"),
        ("DC-only FLaG", "stsb_unfrozen_flag_dconly"),
        ("Exact DC / Mean", "stsb_unfrozen_mean"),
    ]

    rows = []
    for label, exp in conditions:
        m, path = load_metrics(exp, args.seed)
        rows.append({
            "condition": label,
            "test_spearman": float(m["test_spearman"]),
            "test_pearson": float(m["test_pearson"]),
            "best_epoch": int(m["best_epoch"]),
            "metrics_path": str(path),
        })

    full = rows[0]["test_spearman"]
    for row in rows:
        row["relative_to_full"] = (
            row["test_spearman"] / full
            if full != 0 else float("nan")
        )

    out_dir = Path(
        "outputs/text/dc_training_ablation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / (
        f"stsb_unfrozen_dc_ablation_seed{args.seed}.csv"
    )
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print()
    print("=" * 76)
    print(
        f"STSB | UNFROZEN RoBERTa | seed {args.seed} | test metrics"
    )
    print("=" * 76)

    for row in rows:
        print(
            f"{row['condition']:<18s} "
            f"Spearman={row['test_spearman']:.6f} "
            f"({row['relative_to_full']:.1%} of Full) "
            f"Pearson={row['test_pearson']:.6f} "
            f"best_epoch={row['best_epoch']}"
        )

    print()
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
