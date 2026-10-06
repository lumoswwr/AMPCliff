import argparse
import json
from pathlib import Path


def load(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(x):
    return "NA" if x is None else f"{x:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="outputs/text/time_pool_ablation_seed0",
    )
    parser.add_argument(
        "--reference_dir",
        default="outputs/text/author_protocol_5seed",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = Path(args.base_dir)
    ref = Path(args.reference_dir)
    seed = args.seed

    for task in ["stsb", "sprint"]:
        mean_ref = load(
            ref / task / "mean" / f"seed_{seed}" / "metrics.json"
        )

        rows = []
        for time_pool in ["max", "mean"]:
            for method, folder in [
                ("FLaG", "flag"),
                ("Alignment", "alignment"),
            ]:
                m = load(
                    base
                    / time_pool
                    / task
                    / folder
                    / f"seed_{seed}"
                    / "metrics.json"
                )
                rows.append((f"{method}-{time_pool.capitalize()}", m))

        print()
        print("=" * 88)
        print(f"{task.upper()} | time-pool ablation | seed {seed}")
        print("=" * 88)

        if task == "stsb":
            if mean_ref is not None:
                print(
                    f"{'Mean reference':22s} "
                    f"Spearman={fmt(mean_ref.get('test_spearman'))} | "
                    f"Pearson={fmt(mean_ref.get('test_pearson'))}"
                )

            for name, m in rows:
                if m is None:
                    print(f"{name:22s} MISSING")
                    continue

                beta = m.get("mean_anchor_beta")
                suffix = (
                    f" | beta={fmt(beta)}"
                    if beta is not None
                    else ""
                )
                print(
                    f"{name:22s} "
                    f"Spearman={fmt(m.get('test_spearman'))} | "
                    f"Pearson={fmt(m.get('test_pearson'))}"
                    f"{suffix}"
                )

        else:
            if mean_ref is not None:
                print(
                    f"{'Mean reference':22s} "
                    f"Acc={fmt(mean_ref.get('test_accuracy'))} | "
                    f"AP={fmt(mean_ref.get('test_average_precision'))} | "
                    f"F1={fmt(mean_ref.get('test_f1'))}"
                )

            for name, m in rows:
                if m is None:
                    print(f"{name:22s} MISSING")
                    continue

                beta = m.get("mean_anchor_beta")
                suffix = (
                    f" | beta={fmt(beta)}"
                    if beta is not None
                    else ""
                )
                print(
                    f"{name:22s} "
                    f"Acc={fmt(m.get('test_accuracy'))} | "
                    f"AP={fmt(m.get('test_average_precision'))} | "
                    f"F1={fmt(m.get('test_f1'))}"
                    f"{suffix}"
                )


if __name__ == "__main__":
    main()
