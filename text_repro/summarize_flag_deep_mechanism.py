"""Summarize dynamic-vs-static gate, channel-energy and padding probes."""
import argparse
import json
from pathlib import Path

VARIANTS = [("FLaG", "flag"), ("FLaG-B2", "flag_b2"),
            ("FLaG-A1", "flag_a1"), ("FLaG-B1", "flag_b1")]
MODES = [
    "full", "train_mean_gate", "within_batch_swap", "scalar_gate",
    "gate_identity", "imag_equals_real", "no_pad_fixed_gate",
    "long_pad_fixed_gate",
]
DIAG_KEYS = [
    "dc_angle_theoretical_deg",
    "suppressed_real_channel_fraction",
    "suppressed_imag_channel_fraction",
    "suppressed_real_dc_energy_fraction",
    "lowest_5pct_real_gate_dc_energy_fraction",
    "dc_weighted_gate_mean",
    "dc_weighted_gate_sd",
    "real_imag_gate_absolute_difference",
    "dynamic_gate_abs_delta_vs_train_mean",
    "dynamic_gate_l2_delta_vs_train_mean",
    "dynamic_gate_mask_flip_fraction_vs_train_mean",
    "valid_length_fraction",
    "native_to_no_pad_angle_deg",
    "native_to_long_pad_angle_deg",
    "reversal_vs_original_masked_mean_angle_deg",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/text/flag_deep_mechanism_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    args = parser.parse_args()
    root = Path(args.root)
    lines = [
        "# FLaG mechanism deep probe", "",
        f"Seed={args.seed}, split={args.split}", "",
        "All comparisons reuse the same checkpoint. Static gate is calibrated ONLY from TRAIN.",
        "Real/imag same-gate in every frequency bin has an exact time-domain reversal identity.",
        "",
    ]

    for task in ("stsb", "sprint"):
        metric = "spearman" if task == "stsb" else "average_precision"
        results = {}
        for name, folder in VARIANTS:
            path = root / task / folder / f"seed_{args.seed}" / f"deep_{args.split}.json"
            if path.exists():
                results[name] = json.loads(path.read_text())

        print("\n" + "=" * 108)
        print(f"{task.upper()} | {args.split} | {metric}")
        print("=" * 108)
        if not results:
            print("No results.")
            continue

        names = [n for n, _ in VARIANTS if n in results]
        lines.extend([f"## {task.upper()} ({metric})", "",
                      "| Intervention | " + " | ".join(names) + " |",
                      "| --- | " + " | ".join(["---:"]*len(names)) + " |"])
        print(f"{'Intervention':25s}" + "".join(f"{x:>17s}" for x in names))
        for mode in MODES:
            vals = []
            for n in names:
                result = results[n]["mode_performance"][mode]
                val = result[metric]
                vals.append(f"{val:.6f}")
            print(f"{mode:25s}" + "".join(f"{x:>17s}" for x in vals))
            lines.append("| " + mode + " | " + " | ".join(vals) + " |")
        lines += ["", "### Key diagnostics (mean; also inspect JSON percentiles)", "",
                  "| Diagnostic | " + " | ".join(names) + " |",
                  "| --- | " + " | ".join(["---:"]*len(names)) + " |"]
        print("\nDiagnostics:")
        for key in DIAG_KEYS:
            vals = []
            for n in names:
                d = results[n]["diagnostics"].get(key)
                vals.append(f"{d['mean']:.5f}" if d else "NA")
            print(f"{key:52s}" + "".join(f"{x:>17s}" for x in vals))
            lines.append("| " + key + " | " + " | ".join(vals) + " |")
        lines.append("")
    target = root / f"deep_mechanism_summary_{args.split}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {target}")


if __name__ == "__main__":
    main()
