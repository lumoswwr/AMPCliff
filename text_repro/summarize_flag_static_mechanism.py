"""Compare from-scratch trained static mechanism controls to prior FLaG-B2."""
import argparse
import json
from pathlib import Path
import numpy as np

CONTROLS=[
    ("FLaG-B2", "flag_b2", None),
    ("StaticReIm", "static_reim_b2", "new"),
    ("StaticDiag", "static_diag_b2", "new"),
    ("MeanProjection", "mean_proj_b2", "new"),
]


def read(root, task, method, seed):
    path=Path(root)/task/method/f"seed_{seed}"/"metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="outputs/text/flag_static_mechanism")
    p.add_argument("--reference",default="outputs/text/flag_stage_angle_audit_seed0")
    p.add_argument("--reference_fallback",default="outputs/text/flag_init_ablation_seed0")
    p.add_argument("--seeds",default="0")
    args=p.parse_args()
    seeds=[int(x) for x in args.seeds.split()]
    md=["# FLaG vs FFT-free static mechanism trained from scratch","",
        "Matched protocol: time_pool=mean, no post-pool LayerNorm.","",
        "Controls all start as exact Mean and have a trainable identity-initialized projection.",""]
    for task in ("stsb","sprint"):
        metric="test_spearman" if task=="stsb" else "test_average_precision"
        display="Spearman" if task=="stsb" else "Average Precision"
        print("\n"+"="*86)
        print(f"{task.upper()} | {display} | seeds {seeds}")
        print("="*86)
        md.extend([f"## {task.upper()} {display}","",
                   "| Method | n | mean | std | per-seed |",
                   "| --- | ---: | ---: | ---: | --- |"])
        for label, folder, mode in CONTROLS:
            values=[]
            individual=[]
            for seed in seeds:
                if mode is None:
                    m=(read(args.root,task,folder,seed)
                       or read(args.reference,task,folder,seed)
                       or read(args.reference_fallback,task,folder,seed))
                else:
                    m=read(args.root,task,folder,seed)
                if m is not None and metric in m:
                    values.append(float(m[metric]))
                    individual.append(f"s{seed}={m[metric]:.6f}")
            if values:
                avg=float(np.mean(values))
                sd=float(np.std(values,ddof=1)) if len(values)>1 else 0.0
                row=f"{label:24s} n={len(values)} {avg:.6f} ± {sd:.6f}"
                print(row+" | "+"; ".join(individual))
                md.append(f"| {label} | {len(values)} | {avg:.6f} | {sd:.6f} | {'; '.join(individual)} |")
            else:
                print(f"{label:24s} MISSING")
                md.append(f"| {label} | 0 | NA | NA | missing |")
        md.append("")
    path=Path(args.root)/"static_mechanism_summary.md"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("\n".join(md)+"\n",encoding="utf-8")
    print(f"\nSaved {path}")


if __name__=="__main__":
    main()
