#!/usr/bin/env python3
"""Summarize Mean-anchor FLaG seed-0 pilot results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def mean_std(vals):
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def main():
    stsb_root = Path(
        "outputs/text/stsbenchmark/experiments"
    )
    sprint_root = Path(
        "outputs/text/sprintduplicatequestions/experiments"
    )

    print()
    print("=" * 84)
    print("Proposal 1.1 | Mean-anchor orthogonal residual FLaG")
    print("=" * 84)

    stsb_anchor = read(
        stsb_root
        / "stsb_unfrozen_flag_meananchor"
        / "seed_0"
        / "metrics.json"
    )
    stsb_mean = read(
        stsb_root
        / "stsb_unfrozen_mean"
        / "seed_0"
        / "metrics.json"
    )
    stsb_full = read(
        stsb_root
        / "stsb_unfrozen_flag_full"
        / "seed_0"
        / "metrics.json"
    )
    stsb_mr = read(
        stsb_root
        / "stsb_unfrozen_flag_meanresidual"
        / "seed_0"
        / "metrics.json"
    )

    print("\nSTSB | original UNFROZEN protocol | seed 0 | test Spearman")
    print("-" * 74)
    for label, m in [
        ("Mean", stsb_mean),
        ("Original FLaG", stsb_full),
        ("Mean-residual 1.0", stsb_mr),
        ("Mean-anchor 1.1", stsb_anchor),
    ]:
        if m is None:
            print(f"{label:<22s} NA")
            continue

        extra = ""
        if "mean_mix_alpha" in m:
            extra = (
                f" | alpha={float(m['mean_mix_alpha']):.6f}"
            )
        if "mean_anchor_beta" in m:
            extra = (
                f" | beta={float(m['mean_anchor_beta']):.6f}"
            )

        print(
            f"{label:<22s} "
            f"{float(m['test_spearman']):.6f}"
            f"{extra}"
        )

    print("\nSprint | original FROZEN protocol | official test AP")
    print("-" * 74)

    anchor_vals = []
    anchor_betas = []
    full_vals = []

    for seed in [0]:
        a = read(
            sprint_root
            / "sprint_frozen_flag_meananchor"
            / f"seed_{seed}"
            / "metrics.json"
        )
        f = read(
            sprint_root
            / "sprint_frozen_flag"
            / f"seed_{seed}"
            / "metrics.json"
        )

        if a is not None:
            anchor_vals.append(
                float(a["test_average_precision"])
            )
            anchor_betas.append(
                float(a["mean_anchor_beta"])
            )

        if f is not None:
            full_vals.append(
                float(f["test_average_precision"])
            )

    if full_vals:
        m, sd = mean_std(full_vals)
        print(
            f"{'Original FLaG (s0)':<22s} "
            f"{m:.6f} ± {sd:.6f}"
        )

    if anchor_vals:
        m, sd = mean_std(anchor_vals)
        bm, bsd = mean_std(anchor_betas)
        print(
            f"{'Mean-anchor (s0)':<22s} "
            f"{m:.6f} ± {sd:.6f} "
            f"| beta={bm:.6f} ± {bsd:.6f}"
        )

        print("  per-seed:", " ".join(
            f"s{i}={v:.6f}"
            for i, v in enumerate(anchor_vals)
        ))

    frozen_mean = read(
        "outputs/text/dc_training_ablation/"
        "mean_dc_only_metrics.json"
    )
    if frozen_mean is not None:
        print(
            f"{'Mean':<22s} "
            f"{float(frozen_mean['sprint']['test']['cosine_average_precision']):.6f}"
        )

    dc0 = read(
        sprint_root
        / "sprint_frozen_flag_dconly"
        / "seed_0"
        / "metrics.json"
    )
    if dc0 is not None:
        print(
            f"{'DC-only FLaG (s0)':<22s} "
            f"{float(dc0['test_average_precision']):.6f}"
        )


if __name__ == "__main__":
    main()
