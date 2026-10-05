import argparse
import json
from pathlib import Path

import numpy as np


TASKS = {
    "imdb": {
        "metrics": [
            ("test_accuracy", "Accuracy"),
            ("test_f1", "F1"),
        ],
    },
    "stsb": {
        "metrics": [
            ("test_spearman", "Spearman"),
            ("test_pearson", "Pearson"),
        ],
    },
    "sprint": {
        "metrics": [
            ("test_accuracy", "Accuracy"),
            ("test_average_precision", "AP"),
            ("test_f1", "F1"),
        ],
    },
}

METHODS = [
    ("Mean", "mean"),
    ("FLaG", "flag"),
    ("Alignment-aware", "alignment"),
]


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="outputs/text/author_protocol_5seed",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    seeds = [
        int(x.strip())
        for x in args.seeds.split(",")
        if x.strip()
    ]

    full_summary = {}

    for task, spec in TASKS.items():
        print()
        print("=" * 100)
        print(f"{task.upper()} | author protocol | seeds {seeds}")
        print("=" * 100)

        full_summary[task] = {}

        for display, folder in METHODS:
            rows = []
            missing = []

            for seed in seeds:
                path = (
                    base
                    / task
                    / folder
                    / f"seed_{seed}"
                    / "metrics.json"
                )

                if not path.exists():
                    missing.append(seed)
                    continue

                rows.append(
                    json.loads(path.read_text())
                )

            if not rows:
                print(f"{display:18s} NO RESULTS")
                full_summary[task][display] = {
                    "n": 0,
                    "missing_seeds": missing,
                }
                continue

            metric_summary = {}
            pieces = []

            for key, label in spec["metrics"]:
                vals = [row[key] for row in rows]
                mu, sd = mean_std(vals)
                metric_summary[key] = {
                    "mean": mu,
                    "std": sd,
                }
                pieces.append(
                    f"{label}={mu:.6f} ± {sd:.6f}"
                )

            betas = [
                row["mean_anchor_beta"]
                for row in rows
                if "mean_anchor_beta" in row
            ]
            if betas:
                beta_mu, beta_sd = mean_std(betas)
                metric_summary["mean_anchor_beta"] = {
                    "mean": beta_mu,
                    "std": beta_sd,
                }
                pieces.append(
                    f"beta={beta_mu:.6f} ± {beta_sd:.6f}"
                )

            print(
                f"{display:18s} n={len(rows)} | "
                + " | ".join(pieces)
            )

            if missing:
                print(
                    f"{'':18s} missing seeds: {missing}"
                )

            print(
                f"{'':18s} per-seed: "
                + ", ".join(
                    (
                        f"s{row['seed']}="
                        + "/".join(
                            f"{row[key]:.6f}"
                            for key, _ in spec["metrics"]
                        )
                    )
                    for row in rows
                )
            )

            full_summary[task][display] = {
                "n": len(rows),
                "missing_seeds": missing,
                "metrics": metric_summary,
                "per_seed": rows,
            }

    out = base / "summary_5seeds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(full_summary, indent=2)
    )

    print()
    print("Saved:", out)


if __name__ == "__main__":
    main()
