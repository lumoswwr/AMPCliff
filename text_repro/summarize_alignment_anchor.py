#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def read(path):
    p=Path(path)
    if not p.is_file():
        return None
    with p.open() as f:
        return json.load(f)

def fmt(x):
    return "NA" if x is None else f"{float(x):.6f}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args()
    s=args.seed

    sr=Path("outputs/text/stsbenchmark/experiments")
    pr=Path("outputs/text/sprintduplicatequestions/experiments")

    print()
    print("="*90)
    print(f"Proposal 5 | Alignment-aware Mean anchor | seed {s}")
    print("="*90)

    print("\nSTSB | original UNFROZEN protocol | test Spearman")
    print("-"*80)
    for label,name in [
        ("Mean","stsb_unfrozen_mean"),
        ("Original FLaG","stsb_unfrozen_flag_full"),
        ("Mean-residual","stsb_unfrozen_flag_meanresidual"),
        ("Mean-anchor bounded","stsb_unfrozen_flag_meananchor"),
        ("Mean-anchor unbounded","stsb_unfrozen_flag_meananchor_free"),
        ("Alignment-aware anchor","stsb_unfrozen_flag_alignmentanchor"),
    ]:
        m=read(sr/name/f"seed_{s}"/"metrics.json")
        if m is None:
            print(f"{label:<30s} NA")
            continue
        extra=""
        if "mean_mix_alpha" in m:
            extra=f" | alpha={float(m['mean_mix_alpha']):.6f}"
        if "mean_anchor_beta" in m:
            extra=f" | beta={float(m['mean_anchor_beta']):.6f}"
        print(f"{label:<30s} {fmt(m.get('test_spearman'))}{extra}")

    print("\nSprint | original FROZEN protocol | official test AP")
    print("-"*80)
    mean=read("outputs/text/dc_training_ablation/mean_dc_only_metrics.json")
    if mean is not None:
        print(
            f"{'Mean':<30s} "
            f"{float(mean['sprint']['test']['cosine_average_precision']):.6f}"
        )
    for label,name in [
        ("Original FLaG","sprint_frozen_flag"),
        ("DC-only FLaG","sprint_frozen_flag_dconly"),
        ("Mean-residual","sprint_frozen_flag_meanresidual"),
        ("MeanResidual+AttnFreq","sprint_frozen_flag_meanresidual_attnfreq"),
        ("Mean-anchor bounded","sprint_frozen_flag_meananchor"),
        ("Mean-anchor unbounded","sprint_frozen_flag_meananchor_free"),
        ("Alignment-aware anchor","sprint_frozen_flag_alignmentanchor"),
    ]:
        m=read(pr/name/f"seed_{s}"/"metrics.json")
        if m is None:
            print(f"{label:<30s} NA")
            continue
        extra=""
        if "mean_mix_alpha" in m:
            extra=f" | alpha={float(m['mean_mix_alpha']):.6f}"
        if "mean_anchor_beta" in m:
            extra=f" | beta={float(m['mean_anchor_beta']):.6f}"
        print(f"{label:<30s} {fmt(m.get('test_average_precision'))}{extra}")

if __name__=="__main__":
    main()
