import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = [
    ("Mean", "imdb_mean"),
    ("FLaG", "imdb_flag"),
    ("Alignment-aware", "imdb_alignmentanchor"),
]


def parse_seeds(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="outputs/text/imdb/paper_protocol",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    seeds = parse_seeds(args.seeds)

    summary = {}
    rows = []

    print()
    print("=" * 88)
    print("IMDB multi-seed summary")
    print("=" * 88)
    print(
        f"{'Method':18s} "
        f"{'n':>3s}  "
        f"{'Accuracy mean±std':>22s}  "
        f"{'F1 mean±std':>22s}"
    )
    print("-" * 88)

    for display_name, folder in METHODS:
        per_seed = []
        missing = []

        for seed in seeds:
            path = (
                base
                / folder
                / f"seed_{seed}"
                / "metrics.json"
            )

            if not path.exists():
                missing.append(seed)
                continue

            data = json.loads(path.read_text())
            per_seed.append({
                "seed": seed,
                "best_epoch": data.get("best_epoch"),
                "val_accuracy": data.get("val_accuracy"),
                "test_accuracy": data["test_accuracy"],
                "test_f1": data["test_f1"],
                "mean_anchor_beta": data.get("mean_anchor_beta"),
            })

        if not per_seed:
            print(
                f"{display_name:18s} "
                f"{0:3d}  "
                f"{'NO RESULTS':>22s}  "
                f"{'NO RESULTS':>22s}"
            )
            summary[display_name] = {
                "n": 0,
                "missing_seeds": missing,
                "per_seed": [],
            }
            continue

        acc = np.array(
            [x["test_accuracy"] for x in per_seed],
            dtype=float,
        )
        f1 = np.array(
            [x["test_f1"] for x in per_seed],
            dtype=float,
        )

        acc_std = (
            float(acc.std(ddof=1))
            if len(acc) > 1
            else 0.0
        )
        f1_std = (
            float(f1.std(ddof=1))
            if len(f1) > 1
            else 0.0
        )

        acc_mean = float(acc.mean())
        f1_mean = float(f1.mean())

        print(
            f"{display_name:18s} "
            f"{len(per_seed):3d}  "
            f"{acc_mean:.6f} ± {acc_std:.6f}  "
            f"{f1_mean:.6f} ± {f1_std:.6f}"
        )

        summary[display_name] = {
            "n": len(per_seed),
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "missing_seeds": missing,
            "per_seed": per_seed,
        }

        for item in per_seed:
            rows.append({
                "method": display_name,
                **item,
            })

    print("-" * 88)

    for display_name, info in summary.items():
        if info.get("missing_seeds"):
            print(
                f"{display_name}: missing seeds "
                f"{info['missing_seeds']}"
            )

    print()
    print("Per-seed results:")
    for display_name, info in summary.items():
        print(f"\n{display_name}")
        for item in info.get("per_seed", []):
            beta = item.get("mean_anchor_beta")
            beta_text = (
                ""
                if beta is None
                else f" | beta={beta:.6f}"
            )
            print(
                f"  seed {item['seed']}: "
                f"best_epoch={item['best_epoch']} | "
                f"acc={item['test_accuracy']:.6f} | "
                f"f1={item['test_f1']:.6f}"
                f"{beta_text}"
            )

    json_path = base / "imdb_5seed_summary.json"
    csv_path = base / "imdb_5seed_per_seed.csv"

    json_path.write_text(
        json.dumps(summary, indent=2)
    )

    fieldnames = [
        "method",
        "seed",
        "best_epoch",
        "val_accuracy",
        "test_accuracy",
        "test_f1",
        "mean_anchor_beta",
    ]

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Saved:")
    print(" ", json_path)
    print(" ", csv_path)


if __name__ == "__main__":
    main()
