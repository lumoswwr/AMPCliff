#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def read(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args=p.parse_args()
    s=args.seed

    stsb_root=Path("outputs/text/stsbenchmark/experiments")
    sprint_root=Path("outputs/text/sprintduplicatequestions/experiments")

    stsb_rows=[
        ("Mean", read(stsb_root/"stsb_unfrozen_mean"/f"seed_{s}"/"metrics.json")),
        ("Original FLaG", read(stsb_root/"stsb_unfrozen_flag_full"/f"seed_{s}"/"metrics.json")),
        ("DC-only FLaG", read(stsb_root/"stsb_unfrozen_flag_dconly"/f"seed_{s}"/"metrics.json")),
        ("Mean-residual FLaG", read(stsb_root/"stsb_unfrozen_flag_meanresidual"/f"seed_{s}"/"metrics.json")),
        ("Attn-FreqGate FLaG", read(stsb_root/"stsb_unfrozen_flag_attnfreqgate"/f"seed_{s}"/"metrics.json")),
        ("Learned-FreqGate FLaG", read(stsb_root/"stsb_unfrozen_flag_learnedfreqgate"/f"seed_{s}"/"metrics.json")),
    ]

    sprint_rows=[
        ("Original FLaG", read(sprint_root/"sprint_frozen_flag"/f"seed_{s}"/"metrics.json")),
        ("DC-only FLaG", read(sprint_root/"sprint_frozen_flag_dconly"/f"seed_{s}"/"metrics.json")),
        ("Mean-residual FLaG", read(sprint_root/"sprint_frozen_flag_meanresidual"/f"seed_{s}"/"metrics.json")),
        ("Attn-FreqGate FLaG", read(sprint_root/"sprint_frozen_flag_attnfreqgate"/f"seed_{s}"/"metrics.json")),
        ("Learned-FreqGate FLaG", read(sprint_root/"sprint_frozen_flag_learnedfreqgate"/f"seed_{s}"/"metrics.json")),
    ]
    mean_ref=read("outputs/text/dc_training_ablation/mean_dc_only_metrics.json")

    print()
    print("="*86)
    print(f"Proposal 2B | Learned frequency scorer | seed {s}")
    print("="*86)
    print("\nSTSB | original UNFROZEN protocol | test Spearman")
    print("-"*76)
    for label,m in stsb_rows:
        print(f"{label:<26s} " + ("NA" if m is None else f"{float(m['test_spearman']):.6f}"))

    print("\nSprint | original FROZEN protocol | official test AP")
    print("-"*76)
    if mean_ref is not None:
        print(f"{'Mean':<26s} {float(mean_ref['sprint']['test']['cosine_average_precision']):.6f}")
    for label,m in sprint_rows:
        print(f"{label:<26s} " + ("NA" if m is None else f"{float(m['test_average_precision']):.6f}"))

if __name__ == "__main__":
    main()
