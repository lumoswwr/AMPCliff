"""Paired-seed comparison for FLaG, B2, and trainable FFT-free controls.

Use only runs trained by run_flag_paired_ap_multiseed.sh. A prior checkpoint
selected by accuracy must never be mixed with AP-selected runs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

METHOD_DIRS = {
    "FLaG": "flag",
    "FLaG_B2": "flag_b2",
    "MeanProj_B2": "mean_proj_b2",
    "MeanProjRand": "mean_proj_rand",
    "StaticFLaG_ReIm_B2": "static_reim_b2",
    "StaticFLaG_Diag_B2": "static_diag_b2",
}


def read(root, task, method, seed):
    path = root / task / METHOD_DIRS[method] / f"seed_{seed}" / "metrics.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if task == "sprint" and value.get("checkpoint_selection_metric") != "val_average_precision":
        raise ValueError(
            f"{path}: Sprint run is not selected using validation AP; "
            "do not mix models chosen by different validation criteria."
        )
    return value


def fmt(xs):
    if not xs:
        return "NA"
    if len(xs) == 1:
        return f"{float(np.mean(xs)):.6f} (n=1)"
    return f"{float(np.mean(xs)):.6f} ± {float(np.std(xs, ddof=1)):.6f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/text/flag_paired_ap_multiseed")
    p.add_argument("--seeds", default="0 1 2")
    p.add_argument("--tasks", default="sprint stsb")
    args = p.parse_args()
    root = Path(args.root)
    seeds = [int(k) for k in args.seeds.split()]
    tasks = args.tasks.split()
    if not seeds or not tasks:
        raise ValueError("Expected at least one seed and task")
    lines = [
        "# Paired-seed FLaG mechanism comparison", "",
        f"Seeds requested: {', '.join(map(str, seeds))}", "",
        "All models share the same split seed, pooling=mean, no post-pool LayerNorm.",
        "Sprint checkpoints selected by validation **AP**; STSB by validation Spearman.",
        "Sprint RoBERTa frozen; STSB RoBERTa fine-tuned. Do not mix with historical",
        "Sprint accuracy-selected checkpoints.", "",
    ]

    for task in tasks:
        metric = "test_average_precision" if task == "sprint" else "test_spearman"
        validation = "val_average_precision" if task == "sprint" else "val_spearman"
        records = {}
        for method in METHOD_DIRS:
            by_seed = {}
            for seed in seeds:
                value = read(root, task, method, seed)
                if value is not None:
                    by_seed[seed] = value
            records[method] = by_seed
        existing = [method for method in METHOD_DIRS if records[method]]
        if not existing:
            continue

        lines += [f"## {task.upper()} · {metric}", "",
                  "| Model | n | Test mean ± SD | Validation mean ± SD | Epochs |",
                  "| --- | ---: | ---: | ---: | --- |"]
        print("\n" + "="*99)
        print(f"{task.upper()} | {metric} | requested seeds {seeds}")
        print("="*99)
        for method in METHOD_DIRS:
            vals = list(records[method].values())
            if not vals:
                continue
            test = [r[metric] for r in vals]
            val = [r[validation] for r in vals]
            epochs = ", ".join(
                f"s{s}:e{records[method][s]['best_epoch']}"
                for s in sorted(records[method])
            )
            print(f"{method:23s}  n={len(test):2d} test {fmt(test)} val {fmt(val)}  {epochs}")
            lines.append(
                f"| {method} | {len(test)} | {fmt(test)} | {fmt(val)} | {epochs} |"
            )
        lines += ["", "### Strictly paired seed differences in test metric", "",
                  "| Comparison | n paired | Mean paired Δ ± SD | Positive seeds | Individual differences |",
                  "| --- | ---: | ---: | ---: | --- |"]

        compare = [
            ("FLaG", "MeanProjRand"),
            ("MeanProjRand", "MeanProj_B2"),
            ("FLaG", "MeanProj_B2"),
            ("FLaG_B2", "MeanProj_B2"),
            ("FLaG_B2", "StaticFLaG_ReIm_B2"),
            ("StaticFLaG_ReIm_B2", "StaticFLaG_Diag_B2"),
            ("StaticFLaG_Diag_B2", "MeanProj_B2"),
        ]
        for a,b in compare:
            paired = sorted(set(records[a]) & set(records[b]))
            if not paired:
                continue
            differences = [
                records[a][seed][metric] - records[b][seed][metric]
                for seed in paired
            ]
            pos = sum(d > 0 for d in differences)
            delta = fmt(differences)
            note = ", ".join(
                f"s{seed}={diff:+.6f}" for seed,diff in zip(paired, differences)
            )
            print(f"{a} - {b} | n={len(paired)} | Δ={delta} | positive={pos}/{len(paired)}")
            lines.append(
                f"| {a} − {b} | {len(paired)} | {delta} | {pos}/{len(paired)} | {note} |"
            )
        missing = []
        for method in existing:
            for seed in seeds:
                if seed not in records[method]:
                    missing.append(f"{method} s{seed}")
        if missing:
            print("INCOMPLETE:", ", ".join(missing))
            lines += ["", f"**Incomplete model-seed runs:** {', '.join(missing)}"]
        lines += ["", ""]

    path = root / "paired_multiseed_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\nSaved:", path)


if __name__ == "__main__":
    main()
