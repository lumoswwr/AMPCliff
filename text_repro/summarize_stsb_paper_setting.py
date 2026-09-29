#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np

ROOT = Path("outputs/text/stsbenchmark/experiments")

EXPERIMENTS = {
    "FLaG": "stsb_paper_flag_d01_norm1",
    "E12": "stsb_paper_e12_d01_norm1",
}

results = {}

for name, exp in EXPERIMENTS.items():
    rows = []
    print(f"\n{name}  dropout=.1 + norm=True")
    print("-" * 72)

    for seed in range(10):
        path = ROOT / exp / f"seed_{seed}" / "metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)

        with open(path) as f:
            m = json.load(f)

        s = float(m["test_spearman"])
        p = float(m["test_pearson"])
        rows.append((s, p))

        print(
            f"seed {seed}: "
            f"Spearman={s:.9f}  "
            f"Pearson={p:.9f}"
        )

    arr = np.asarray(rows, dtype=float)
    results[name] = arr

    print(
        f"MEAN:   "
        f"Spearman={arr[:,0].mean():.9f} ± {arr[:,0].std(ddof=1):.9f}  "
        f"Pearson={arr[:,1].mean():.9f} ± {arr[:,1].std(ddof=1):.9f}"
    )

delta = results["E12"] - results["FLaG"]

print("\nE12 - FLaG paired delta")
print("-" * 72)

for seed in range(10):
    print(
        f"seed {seed}: "
        f"ΔSpearman={delta[seed,0]:+.9f}  "
        f"ΔPearson={delta[seed,1]:+.9f}"
    )

print(
    f"MEAN:   "
    f"ΔSpearman={delta[:,0].mean():+.9f} ± {delta[:,0].std(ddof=1):.9f}  "
    f"ΔPearson={delta[:,1].mean():+.9f} ± {delta[:,1].std(ddof=1):.9f}"
)

print(
    "positive Spearman seeds:",
    f"{int((delta[:,0] > 0).sum())}/10"
)
print(
    "positive Pearson seeds:",
    f"{int((delta[:,1] > 0).sum())}/10"
)
