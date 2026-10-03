#!/usr/bin/env python3
"""Summarize proposal-1 Mean-residual FLaG against existing controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def fmt(v):
    return "NA" if v is None else f"{v:.6f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    s = args.seed

    stsb_root = Path(
        "outputs/text/stsbenchmark/experiments"
    )
    sprint_root = Path(
        "outputs/text/sprintduplicatequestions/experiments"
    )

    stsb_full = read(
        stsb_root / "stsb_unfrozen_flag_full"
        / f"seed_{s}" / "metrics.json"
    )
    stsb_dc = read(
        stsb_root / "stsb_unfrozen_flag_dconly"
        / f"seed_{s}" / "metrics.json"
    )
    stsb_mean = read(
        stsb_root / "stsb_unfrozen_mean"
        / f"seed_{s}" / "metrics.json"
    )
    stsb_mr = read(
        stsb_root / "stsb_unfrozen_flag_meanresidual"
        / f"seed_{s}" / "metrics.json"
    )

    sprint_full = read(
        sprint_root / "sprint_frozen_flag"
        / f"seed_{s}" / "metrics.json"
    )
    sprint_dc = read(
        sprint_root / "sprint_frozen_flag_dconly"
        / f"seed_{s}" / "metrics.json"
    )
    sprint_mr = read(
        sprint_root / "sprint_frozen_flag_meanresidual"
        / f"seed_{s}" / "metrics.json"
    )
    frozen_mean = read(
        "outputs/text/dc_training_ablation/"
        "mean_dc_only_metrics.json"
    )

    print()
    print("=" * 82)
    print(f"Proposal 1 | Mean-residual FLaG | seed {s}")
    print("=" * 82)

    print("\nSTSB | original UNFROZEN protocol | test Spearman")
    print("-" * 68)
    stsb_rows = [
        ("Mean", stsb_mean),
        ("Original FLaG", stsb_full),
        ("DC-only FLaG", stsb_dc),
        ("Mean-residual FLaG", stsb_mr),
    ]
    for label, m in stsb_rows:
        val = None if m is None else float(m["test_spearman"])
        extra = ""
        if m is not None and "mean_mix_alpha" in m:
            extra = f" | alpha={float(m['mean_mix_alpha']):.6f}"
        print(f"{label:<22s} {fmt(val)}{extra}")

    print("\nSprint | original FROZEN protocol | official test AP")
    print("-" * 68)
    sprint_mean = (
        None
        if frozen_mean is None
        else float(
            frozen_mean["sprint"]["test"][
                "cosine_average_precision"
            ]
        )
    )
    sprint_rows = [
        ("Mean", sprint_mean, None),
        (
            "Original FLaG",
            None if sprint_full is None
            else float(sprint_full["test_average_precision"]),
            sprint_full,
        ),
        (
            "DC-only FLaG",
            None if sprint_dc is None
            else float(sprint_dc["test_average_precision"]),
            sprint_dc,
        ),
        (
            "Mean-residual FLaG",
            None if sprint_mr is None
            else float(sprint_mr["test_average_precision"]),
            sprint_mr,
        ),
    ]
    for label, val, m in sprint_rows:
        extra = ""
        if m is not None and "mean_mix_alpha" in m:
            extra = f" | alpha={float(m['mean_mix_alpha']):.6f}"
        print(f"{label:<22s} {fmt(val)}{extra}")

    if sprint_mr is not None:
        print("\nSprint Mean-residual validation AP:",
              f"{float(sprint_mr['final_val_average_precision']):.6f}")


if __name__ == "__main__":
    main()
