#!/usr/bin/env python3
"""Summarize the matched Full / No-DC / DC-only FLaG experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as f:
        return json.load(f)


def ratio(x, ref):
    return float(x / ref) if ref else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/text/dc_training_ablation"),
    )
    args = ap.parse_args()

    s = args.seed

    stsb_full = load_json(Path(
        f"outputs/text/stsbenchmark/experiments/"
        f"stsb_frozen_flag/seed_{s}/metrics.json"
    ))
    stsb_nodc = load_json(Path(
        f"outputs/text/stsbenchmark/experiments/"
        f"stsb_frozen_flag_nodc/seed_{s}/metrics.json"
    ))
    stsb_dconly = load_json(Path(
        f"outputs/text/stsbenchmark/experiments/"
        f"stsb_frozen_flag_dconly/seed_{s}/metrics.json"
    ))

    sprint_full = load_json(Path(
        f"outputs/text/sprintduplicatequestions/experiments/"
        f"sprint_frozen_flag/seed_{s}/metrics.json"
    ))
    sprint_nodc = load_json(Path(
        f"outputs/text/sprintduplicatequestions/experiments/"
        f"sprint_frozen_flag_nodc/seed_{s}/metrics.json"
    ))
    sprint_dconly = load_json(Path(
        f"outputs/text/sprintduplicatequestions/experiments/"
        f"sprint_frozen_flag_dconly/seed_{s}/metrics.json"
    ))

    mean = load_json(
        args.output_dir / "mean_dc_only_metrics.json"
    )

    stsb_full_v = float(stsb_full["test_spearman"])
    stsb_nodc_v = float(stsb_nodc["test_spearman"])
    stsb_dconly_v = float(stsb_dconly["test_spearman"])
    stsb_mean_v = float(
        mean["stsb"]["test"]["spearman"]
    )

    sprint_full_v = float(
        sprint_full["final_val_average_precision"]
    )
    sprint_nodc_v = float(
        sprint_nodc["final_val_average_precision"]
    )
    sprint_dconly_v = float(
        sprint_dconly["final_val_average_precision"]
    )
    sprint_mean_v = float(
        mean["sprint"]["validation"][
            "cosine_average_precision"
        ]
    )

    rows = [
        {
            "task": "STSB",
            "metric": "test Spearman",
            "full_flag": stsb_full_v,
            "no_dc_flag": stsb_nodc_v,
            "dc_only_flag": stsb_dconly_v,
            "exact_dc_mean": stsb_mean_v,
            "no_dc_over_full": ratio(stsb_nodc_v, stsb_full_v),
            "dc_only_flag_over_full": ratio(
                stsb_dconly_v, stsb_full_v
            ),
            "mean_over_full": ratio(
                stsb_mean_v, stsb_full_v
            ),
        },
        {
            "task": "Sprint",
            "metric": "validation AP",
            "full_flag": sprint_full_v,
            "no_dc_flag": sprint_nodc_v,
            "dc_only_flag": sprint_dconly_v,
            "exact_dc_mean": sprint_mean_v,
            "no_dc_over_full": ratio(
                sprint_nodc_v, sprint_full_v
            ),
            "dc_only_flag_over_full": ratio(
                sprint_dconly_v, sprint_full_v
            ),
            "mean_over_full": ratio(
                sprint_mean_v, sprint_full_v
            ),
        },
    ]

    df = pd.DataFrame(rows)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    out_csv = (
        args.output_dir
        / f"matched_dc_ablation_seed{s}.csv"
    )
    df.to_csv(out_csv, index=False)

    print()
    print("=" * 96)
    print(
        "Matched frozen-backbone spectral ablation "
        f"| seed {s}"
    )
    print("=" * 96)

    for row in rows:
        print()
        print(
            f"{row['task']} | {row['metric']}"
        )
        print("-" * 72)
        print(
            f"Full FLaG       : {row['full_flag']:.6f}"
        )
        print(
            f"No-DC FLaG      : {row['no_dc_flag']:.6f} "
            f"({row['no_dc_over_full']:.1%} of Full)"
        )
        print(
            f"DC-only FLaG    : {row['dc_only_flag']:.6f} "
            f"({row['dc_only_flag_over_full']:.1%} of Full)"
        )
        print(
            f"Exact DC / Mean : {row['exact_dc_mean']:.6f} "
            f"({row['mean_over_full']:.1%} of Full)"
        )

    print()
    print("Primary fair comparison:")
    print(
        "  Full / No-DC / DC-only FLaG use the same frozen "
        "backbone and the same trainable FLaG capacity."
    )
    print(
        "  Mean is an additional parameter-free exact-DC readout, "
        "not a capacity-matched model."
    )
    print()
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
