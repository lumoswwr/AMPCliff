import argparse
import json
from pathlib import Path


def load(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt(value):
    return "NA" if value is None else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="outputs/text/flag_init_ablation_seed0",
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

    variants = [
        ("FLaG-Mean", "flag"),
        ("FLaG-A1", "flag_a1"),
        ("FLaG-B1", "flag_b1"),
        ("FLaG-zero/B2", "flag_zero"),
    ]

    for task in ["stsb", "sprint"]:
        mean_ref = load(
            ref / task / "mean" / f"seed_{seed}" / "metrics.json"
        )

        print()
        print("=" * 92)
        print(f"{task.upper()} | FLaG initialization ablation | seed {seed}")
        print("=" * 92)

        if task == "stsb":
            if mean_ref is not None:
                print(
                    f"{'Mean reference':18s} "
                    f"Spearman={fmt(mean_ref.get('test_spearman'))} | "
                    f"Pearson={fmt(mean_ref.get('test_pearson'))}"
                )

            for name, folder in variants:
                m = load(
                    base
                    / task
                    / folder
                    / f"seed_{seed}"
                    / "metrics.json"
                )
                if m is None:
                    print(f"{name:18s} MISSING")
                    continue
                print(
                    f"{name:18s} "
                    f"Spearman={fmt(m.get('test_spearman'))} | "
                    f"Pearson={fmt(m.get('test_pearson'))} | "
                    f"best_epoch={m.get('best_epoch')}"
                )
        else:
            if mean_ref is not None:
                print(
                    f"{'Mean reference':18s} "
                    f"Acc={fmt(mean_ref.get('test_accuracy'))} | "
                    f"AP={fmt(mean_ref.get('test_average_precision'))} | "
                    f"F1={fmt(mean_ref.get('test_f1'))}"
                )

            for name, folder in variants:
                m = load(
                    base
                    / task
                    / folder
                    / f"seed_{seed}"
                    / "metrics.json"
                )
                if m is None:
                    print(f"{name:18s} MISSING")
                    continue
                print(
                    f"{name:18s} "
                    f"Acc={fmt(m.get('test_accuracy'))} | "
                    f"AP={fmt(m.get('test_average_precision'))} | "
                    f"F1={fmt(m.get('test_f1'))} | "
                    f"best_epoch={m.get('best_epoch')}"
                )


if __name__ == "__main__":
    main()
