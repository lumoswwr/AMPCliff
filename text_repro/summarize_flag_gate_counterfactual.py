"""Summarize trained-checkpoint gate interventions."""
import argparse
import json
from pathlib import Path

METHODS = [
    ("FLaG", "flag"), ("FLaG-A1", "flag_a1"),
    ("FLaG-B1", "flag_b1"), ("FLaG-B2", "flag_b2"),
]
MODES = [
    "full", "dc_identity", "ac_identity", "gate_identity", "scalar_gate",
    "bias_only", "feature_only", "no_projection", "raw_mean",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--root", default="outputs/text/flag_gate_counterfactual_seed0")
    args = p.parse_args()
    root = Path(args.root)
    lines = ["# FLaG Gate Mechanism: Checkpoint Interventions", "",
             f"Seed {args.seed}; split {args.split}", "",
             "All rows share a trained checkpoint and a fixed cosine head.", ""]

    for task in ["stsb", "sprint"]:
        key = "spearman" if task == "stsb" else "average_precision"
        records = {}
        for name, folder in METHODS:
            f = root / task / folder / f"seed_{args.seed}" / f"probe_{args.split}.json"
            if f.is_file():
                records[name] = json.loads(f.read_text())

        print("\n" + "="*96)
        print(f"{task.upper()} | {args.split} | {key}")
        print("="*96)
        methods = [name for name, _folder in METHODS if name in records]
        if not methods:
            print("No completed results")
            continue
        lines += [
            f"## {task.upper()} ({key})", "",
            "| Intervention | " + " | ".join(methods) + " |",
            "| --- | " + " | ".join(["---:"]*len(methods)) + " |",
        ]
        for name in methods:
            r = records[name]
            print(f"{name}: n={r['n_pairs']}" + (
                f", prevalence={r['positive_prevalence']:.4f}"
                if task == "sprint" else ""
            ))
        print(f"{'Mode':19s}" + "".join(f"{name:>17s}" for name in methods))
        for mode in MODES:
            vals = [
                f"{records[name]['mode_results'][mode][key]:.6f}"
                for name in methods
            ]
            print(f"{mode:19s}" + "".join(f"{v:>17s}" for v in vals))
            lines.append("| " + mode + " | " + " | ".join(vals) + " |")
        lines += ["", "### Gate internal diagnostics", ""]
        for name in methods:
            d = records[name]["gate_diagnostics"]
            keys = [
                "dc_angle_deg", "gate_std_across_channels",
                "gate_cv_across_channels", "gate_fraction_below_1",
                "sigmoid_fraction_below_.05", "sigmoid_fraction_above_.95",
                "gate_input_channel_sd", "W1_channel_sd", "GELU_channel_sd",
                "logits_channel_sd", "logit_between_sentence_sd",
            ]
            v = {key: round(d[key]["mean"], 5) for key in keys if key in d}
            print(name, json.dumps(v))
            lines.append(f"- {name}: " + json.dumps(v))
        lines.append("")

    target = root / f"counterfactual_summary_{args.split}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {target}")


if __name__ == "__main__":
    main()
