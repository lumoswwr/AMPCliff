#!/usr/bin/env python3
"""Summarize Proposal-2 attention-derived frequency-wise FLaG."""

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

    stsb = {
        "Mean": read(
            stsb_root / "stsb_unfrozen_mean"
            / f"seed_{s}" / "metrics.json"
        ),
        "Original FLaG": read(
            stsb_root / "stsb_unfrozen_flag_full"
            / f"seed_{s}" / "metrics.json"
        ),
        "DC-only FLaG": read(
            stsb_root / "stsb_unfrozen_flag_dconly"
            / f"seed_{s}" / "metrics.json"
        ),
        "Attn-FreqGate FLaG": read(
            stsb_root / "stsb_unfrozen_flag_attnfreqgate"
            / f"seed_{s}" / "metrics.json"
        ),
    }

    sprint = {
        "Original FLaG": read(
            sprint_root / "sprint_frozen_flag"
            / f"seed_{s}" / "metrics.json"
        ),
        "DC-only FLaG": read(
            sprint_root / "sprint_frozen_flag_dconly"
            / f"seed_{s}" / "metrics.json"
        ),
        "Attn-FreqGate FLaG": read(
            sprint_root / "sprint_frozen_flag_attnfreqgate"
            / f"seed_{s}" / "metrics.json"
        ),
    }

    mean_ref = read(
        "outputs/text/dc_training_ablation/"
        "mean_dc_only_metrics.json"
    )

    print()
    print("=" * 82)
    print(
        f"Proposal 2 | Attention-derived frequency gate | seed {s}"
    )
    print("=" * 82)

    print("\nSTSB | original UNFROZEN protocol | test Spearman")
    print("-" * 72)
    for label, m in stsb.items():
        if m is None:
            print(f"{label:<24s} NA")
        else:
            print(
                f"{label:<24s} "
                f"{float(m['test_spearman']):.6f}"
            )

    print("\nSprint | original FROZEN protocol | official test AP")
    print("-" * 72)

    if mean_ref is not None:
        print(
            f"{'Mean':<24s} "
            f"{float(mean_ref['sprint']['test']['cosine_average_precision']):.6f}"
        )

    for label, m in sprint.items():
        if m is None:
            print(f"{label:<24s} NA")
        else:
            print(
                f"{label:<24s} "
                f"{float(m['test_average_precision']):.6f}"
            )

    candidate = sprint["Attn-FreqGate FLaG"]
    if candidate is not None:
        print(
            "\nSprint candidate validation AP:",
            f"{float(candidate['final_val_average_precision']):.6f}",
        )


if __name__ == "__main__":
    main()
