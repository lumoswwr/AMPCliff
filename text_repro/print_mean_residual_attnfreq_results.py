#!/usr/bin/env python3
"""Print final results for Mean-residual + Attn-FreqGate experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_json(path: Path):
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def fmt(x):
    return "NA" if x is None else f"{float(x):.6f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    s = args.seed

    stsb_root = Path("outputs/text/stsbenchmark/experiments")
    sprint_root = Path("outputs/text/sprintduplicatequestions/experiments")
    sweep_root = Path(
        "outputs/text/mean_residual_attnfreq_alpha_sweep"
    )

    stsb = read_json(
        stsb_root
        / "stsb_unfrozen_flag_meanresidual_attnfreq"
        / f"seed_{s}"
        / "metrics.json"
    )
    sprint = read_json(
        sprint_root
        / "sprint_frozen_flag_meanresidual_attnfreq"
        / f"seed_{s}"
        / "metrics.json"
    )

    print()
    print("=" * 86)
    print(
        f"Mean-residual + Attn-FreqGate | seed {s} | final results"
    )
    print("=" * 86)

    print("\nSTSB | original UNFROZEN protocol")
    print("-" * 72)
    if stsb is None:
        print("metrics.json not found")
    else:
        print(
            f"test Spearman        {fmt(stsb.get('test_spearman'))}"
        )
        print(
            f"test Pearson         {fmt(stsb.get('test_pearson'))}"
        )
        print(
            f"best epoch           {stsb.get('best_epoch', 'NA')}"
        )
        print(
            f"learned alpha        {fmt(stsb.get('mean_mix_alpha'))}"
        )

    print("\nSprint | original FROZEN protocol")
    print("-" * 72)
    if sprint is None:
        print("metrics.json not found")
    else:
        print(
            f"test AP              {fmt(sprint.get('test_average_precision'))}"
        )
        print(
            f"test cosine AP       {fmt(sprint.get('test_cosine_average_precision'))}"
        )
        print(
            f"final val AP         {fmt(sprint.get('final_val_average_precision'))}"
        )
        print(
            f"best epoch           {sprint.get('best_epoch', 'NA')}"
        )
        print(
            f"learned alpha        {fmt(sprint.get('mean_mix_alpha'))}"
        )

    print("\nValidation-only alpha sweep")
    print("-" * 72)

    stsb_csv = sweep_root / f"stsb_alpha_sweep_seed{s}.csv"
    if stsb_csv.is_file():
        df = pd.read_csv(stsb_csv)
        row = df.loc[df["val_spearman"].idxmax()]
        print(
            "STSB best alpha       "
            f"{float(row['alpha']):.2f} | "
            f"val Spearman={float(row['val_spearman']):.6f}"
        )
    else:
        print("STSB sweep CSV       NA")

    sprint_csv = sweep_root / f"sprint_alpha_sweep_seed{s}.csv"
    if sprint_csv.is_file():
        df = pd.read_csv(sprint_csv)
        row = df.loc[df["val_ap"].idxmax()]
        print(
            "Sprint best alpha     "
            f"{float(row['alpha']):.2f} | "
            f"val AP={float(row['val_ap']):.6f}"
        )
    else:
        print("Sprint sweep CSV     NA")

    print()


if __name__ == "__main__":
    main()
