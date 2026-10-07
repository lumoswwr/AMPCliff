import argparse
import json
from pathlib import Path


VARIANTS = [
    ("FLaG-Mean", "flag"),
    ("FLaG-A1", "flag_a1"),
    ("FLaG-B1", "flag_b1"),
    ("FLaG-zero/B2", "flag_b2"),
]

ANGLE_KEYS = [
    ("Mean->DC", "mean_to_dc_deg"),
    ("SpecGate", "spectral_gate_deg"),
    ("DCGate", "dc_gate_deg"),
    ("Mean->TimeMean", "mean_to_time_mean_deg"),
    ("Time->PostNorm", "time_mean_to_postnorm_deg"),
    ("PostNorm->Proj", "postnorm_to_projection_deg"),
    ("Mean->Final", "mean_to_final_deg"),
]


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def angle_mean(snapshot, key):
    if snapshot is None:
        return None
    value = snapshot.get(key)
    if not isinstance(value, dict):
        return None
    return value.get("mean")


def fmt(value, digits=3):
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def metric_line(task, metrics):
    if metrics is None:
        return "MISSING"
    if task == "stsb":
        return (
            f"Spearman={metrics.get('test_spearman', float('nan')):.6f} | "
            f"Pearson={metrics.get('test_pearson', float('nan')):.6f} | "
            f"best_epoch={metrics.get('best_epoch')}"
        )
    return (
        f"Acc={metrics.get('test_accuracy', float('nan')):.6f} | "
        f"AP={metrics.get('test_average_precision', float('nan')):.6f} | "
        f"F1={metrics.get('test_f1', float('nan')):.6f} | "
        f"best_epoch={metrics.get('best_epoch')}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default="outputs/text/flag_stage_angle_audit_seed0",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = Path(args.base_dir)
    seed = args.seed
    md_lines = [
        "# FLaG stage-wise angle audit",
        "",
        f"seed: {seed}",
        "",
    ]

    for task in ["stsb", "sprint"]:
        print()
        print("=" * 110)
        print(f"{task.upper()} | FLaG stage-wise angle audit | seed {seed}")
        print("=" * 110)

        md_lines += [
            f"## {task.upper()}",
            "",
            "### Performance",
            "",
            "| Variant | Result |",
            "| --- | --- |",
        ]

        data = {}
        for label, folder in VARIANTS:
            run_dir = (
                base
                / task
                / folder
                / f"seed_{seed}"
            )
            metrics = load_json(run_dir / "metrics.json")
            angles = load_json(run_dir / "stage_angles.json")
            data[label] = (metrics, angles)

            line = metric_line(task, metrics)
            print(f"{label:18s} {line}")
            md_lines.append(f"| {label} | {line} |")

        print()
        header = (
            f"{'Variant':18s} {'Checkpoint':10s} "
            f"{'Mean->DC':>9s} {'SpecGate':>9s} {'DCGate':>9s} "
            f"{'Mean->Time':>11s} {'Time->Norm':>11s} "
            f"{'Norm->Proj':>11s} {'Mean->Final':>12s}"
        )
        print(header)
        print("-" * len(header))

        md_lines += [
            "",
            "### Angles (degrees, validation-set mean)",
            "",
            "| Variant | Checkpoint | Mean->DC | Spectral gate | DC gate | Mean->time mean | Time mean->post norm | Post norm->projection | Mean->final |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]

        for label, _folder in VARIANTS:
            metrics, audit = data[label]
            for checkpoint in ["epoch0", "epoch1", "best"]:
                snapshot = None if audit is None else audit.get(checkpoint)
                if snapshot is None:
                    vals = [None] * len(ANGLE_KEYS)
                else:
                    vals = [
                        angle_mean(snapshot, key)
                        for _name, key in ANGLE_KEYS
                    ]

                ckpt_label = checkpoint
                if checkpoint == "best" and audit is not None:
                    ckpt_label = f"best(e{audit.get('best_epoch', '?')})"

                print(
                    f"{label:18s} {ckpt_label:10s} "
                    f"{fmt(vals[0]):>9s} {fmt(vals[1]):>9s} {fmt(vals[2]):>9s} "
                    f"{fmt(vals[3]):>11s} {fmt(vals[4]):>11s} "
                    f"{fmt(vals[5]):>11s} {fmt(vals[6]):>12s}"
                )

                md_lines.append(
                    "| "
                    + " | ".join(
                        [
                            label,
                            ckpt_label,
                            *[fmt(v) for v in vals],
                        ]
                    )
                    + " |"
                )

        md_lines.append("")

    out_path = base / "stage_angle_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )
    print()
    print(f"Saved markdown summary: {out_path}")


if __name__ == "__main__":
    main()
