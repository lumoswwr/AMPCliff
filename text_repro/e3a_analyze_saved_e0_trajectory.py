#!/usr/bin/env python3
"""E3a: read-only analysis of SAVED E0 STSB validation trajectory and pairs.

No model load, GPU, training, parameter updates, or test-set access.
Outputs (fixed filenames, no overwrite) are only created in full mode:
  outputs/text/stsb_mean_equiv_e3_trajectory/seed_N/e3_trajectory_analysis.json
  outputs/text/stsb_mean_equiv_e3_trajectory/seed_N/e3_val_mse.png
  outputs/text/stsb_mean_equiv_e3_trajectory/seed_N/e3_val_spearman.png

Only previously SAVED E0 probe steps can be diagnosed. A crossing between
sparse probes is not evidence of the exact intervening optimizer step.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr


METRIC_TOL = {"mse": 1e-6, "spearman": 1e-6, "pearson": 1e-6}
REQUIRED_ARRAY_KEYS = (
    "labels", "score_full", "score_same_encoder_mean",
    "cos_full", "cos_same_encoder_mean",
)


def measure(labels, scores):
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    y = labels / 5.0
    return {
        "mse": float(np.mean((scores - y) ** 2)),
        "spearman": float(spearmanr(labels, scores).statistic),
        "pearson": float(pearsonr(labels, scores).statistic),
    }


def load_npz(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing E0 saved pair predictions: {path}; "
            "do not infer unrecorded steps.")
    with np.load(path, allow_pickle=False) as raw:
        if not set(REQUIRED_ARRAY_KEYS).issubset(raw.files):
            missing = sorted(set(REQUIRED_ARRAY_KEYS) - set(raw.files))
            raise RuntimeError(f"Missing NPZ arrays {missing}: {path}")
        arrays = {key: np.asarray(raw[key]).copy()
                  for key in REQUIRED_ARRAY_KEYS}
    lengths = {len(v) for v in arrays.values()}
    if len(lengths) != 1 or not all(v.ndim == 1 for v in arrays.values()):
        raise RuntimeError(f"Bad array shape(s): {path}")
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise RuntimeError(f"NaN/Inf in arrays: {path}")
    return arrays


def verify_metrics(measured, recorded, description):
    for metric, tolerance in METRIC_TOL.items():
        got, expected = measured[metric], float(recorded[metric])
        if not np.isfinite(got) or abs(got - expected) > tolerance:
            raise RuntimeError(
                f"{description} {metric}: recalculated {got:.9f} "
                f"does not reproduce E0 {expected:.9f}. STOP.")


def first_matching(rows, condition):
    for row in rows:
        if condition(row):
            return {
                "step": row["step"],
                "epoch": row["epoch"],
                "phase": row["phase"],
            }
    return None


def summary_events(rows):
    start = rows[0]
    baseline_mse = start["validation"]["mse"]
    baseline_spearman = start["validation"]["spearman"]
    eps_mse = 1e-8
    eps_sp = 1e-7
    return {
        "first_observed_validation_mse_above_step0": first_matching(
            rows[1:], lambda r: r["validation"]["mse"]
            > baseline_mse + eps_mse),
        "first_observed_validation_spearman_below_step0": first_matching(
            rows[1:], lambda r: r["validation"]["spearman"]
            < baseline_spearman - eps_sp),
        "first_observed_validation_mse_increase_from_previous_probe":
            first_matching(
                rows[1:], lambda r: r["delta_from_previous_probe"]["mse"]
                > eps_mse),
        "first_observed_validation_spearman_decrease_from_previous_probe":
            first_matching(
                rows[1:], lambda r: r["delta_from_previous_probe"]["spearman"]
                < -eps_sp),
        "best_observed_validation_mse_step": min(
            rows, key=lambda r: r["validation"]["mse"])["step"],
        "best_observed_validation_spearman_step": max(
            rows, key=lambda r: r["validation"]["spearman"])["step"],
        "final_validation_mse_change_vs_step0":
            rows[-1]["validation"]["mse"] - baseline_mse,
        "final_validation_spearman_change_vs_step0":
            rows[-1]["validation"]["spearman"] - baseline_spearman,
        "meaning": (
            "Earliest OBSERVED probe only. Any real crossover may occur "
            "between checkpoints. No claim of monotonic behavior "
            "between observations."
        ),
    }


def analyze(pilot, seed):
    summary_path = pilot / "summary.json"
    trajectory_path = pilot / "trajectory.jsonl"
    if not summary_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError(
            "E3a requires existing E0 summary.json and trajectory.jsonl "
            f"under {pilot}; it never reruns the model.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["seed"]) != seed or summary.get("frozen_roberta") is not True:
        raise RuntimeError("E0 seed/frozen backbone mismatch.")
    original = [json.loads(line) for line in trajectory_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if not original:
        raise RuntimeError("E0 trajectory is empty.")

    steps = [int(r["global_step"]) for r in original]
    if steps[0] != 0 or len(steps) != len(set(steps)):
        raise RuntimeError("Expected unique E0 probes including step 0.")
    if steps != sorted(steps):
        raise RuntimeError("E0 trajectory probe steps are not sorted.")
    if steps[-1] != int(summary["steps"]):
        raise RuntimeError("Last E0 probe is not its recorded final step.")

    base_arrays = load_npz(pilot / "validation_pairs_step_0.npz")
    base_labels = base_arrays["labels"]
    if len(base_labels) != int(summary["n_validation"]):
        raise RuntimeError("Wrong validation set size in original E0 probes.")
    base_ranks = rankdata(base_arrays["cos_full"].astype(np.float64))
    base_error_sq = (
        base_arrays["score_full"].astype(np.float64) -
        base_labels.astype(np.float64) / 5.0
    ) ** 2

    rows = []
    for i, raw in enumerate(original):
        step = int(raw["global_step"])
        arrays = load_npz(pilot / f"validation_pairs_step_{step}.npz")
        if not np.array_equal(arrays["labels"], base_labels):
            raise RuntimeError(
                f"Different validation label order at E0 step {step}.")
        full = measure(base_labels, arrays["score_full"])
        mean_with_head = measure(
            base_labels, arrays["score_same_encoder_mean"])
        verify_metrics(full, raw["full_flag_b2"], f"Step {step} B2")
        verify_metrics(mean_with_head,
                       raw["same_encoder_mean_with_current_head"],
                       f"Step {step} Mean+current-head")
        cos = arrays["cos_full"].astype(np.float64)
        mean_cos = arrays["cos_same_encoder_mean"].astype(np.float64)
        if np.max(np.abs(cos - mean_cos)) > 2.0 + 1e-5:
            raise RuntimeError(f"Impossible cosine difference at step {step}.")
        ranks = rankdata(cos)
        rank_delta = np.abs(ranks - base_ranks)
        cur_sqerr = (arrays["score_full"].astype(np.float64) -
                     base_labels.astype(np.float64) / 5.0) ** 2
        improve = (base_error_sq - cur_sqerr)
        prev = rows[-1] if rows else None

        row = {
            "step": step,
            "epoch": int(raw["epoch"]),
            "phase": raw["phase"],
            "validation": full,
            "mean_pool_with_current_head_validation": mean_with_head,
            "delta_vs_step0": {
                metric: full[metric] -
                (rows[0]["validation"][metric] if rows else full[metric])
                for metric in METRIC_TOL
            },
            "delta_from_previous_probe": {
                metric: full[metric] -
                (prev["validation"][metric] if prev else full[metric])
                for metric in METRIC_TOL
            },
            "pooling_mse_contribution_vs_mean_with_same_head":
                full["mse"] - mean_with_head["mse"],
            "mean_absolute_pair_cos_change_from_step0":
                float(np.abs(cos - base_arrays["cos_full"]).mean()),
            "max_absolute_pair_cos_change_from_step0":
                float(np.abs(cos - base_arrays["cos_full"]).max()),
            "mean_absolute_pair_cos_difference_from_mean":
                float(np.abs(cos - mean_cos).mean()),
            "pair_rank_movement_from_step0": {
                "mean_abs_rank_delta": float(rank_delta.mean()),
                "max_abs_rank_delta": float(rank_delta.max()),
                "fraction_moved_at_least_10_ranks":
                    float((rank_delta >= 10).mean()),
                "fraction_moved_at_least_50_ranks":
                    float((rank_delta >= 50).mean()),
                "number_with_changed_rank": int((rank_delta > 0).sum()),
            },
            "per_pair_squared_error_vs_step0": {
                "improved_pairs": int((improve > 1e-9).sum()),
                "worsened_pairs": int((improve < -1e-9).sum()),
                "essentially_unchanged_pairs":
                    int((np.abs(improve) <= 1e-9).sum()),
            },
            "last_online_minibatch_train_mse":
                raw.get("last_minibatch_train_mse"),
            "head_scale": float(raw["head_scale"]),
            "head_bias": float(raw["head_bias"]),
        }
        rows.append(row)

    first = rows[0]
    if abs(first["validation"]["spearman"] -
           float(summary["mean_reference_validation"]["spearman"])) > 1e-6:
        raise RuntimeError("Step-0 Mean reference Spearman mismatch.")
    if abs(rows[-1]["validation"]["spearman"] -
           float(summary["final_b2_val_spearman"])) > 1e-6:
        raise RuntimeError("E0 final Spearman mismatch.")
    if any(abs(r["mean_pool_with_current_head_validation"]["spearman"] -
               first["validation"]["spearman"]) > 2e-6 for r in rows):
        raise RuntimeError(
            "Same frozen RoBERTa Mean changed rank across probes; "
            "the positive affine head should not affect Spearman.")

    return {
        "experiment": "E3a_saved_E0_validation_trajectory",
        "seed": seed,
        "read_only_existing_e0": True,
        "original_e0_summary": str(summary_path),
        "original_e0_trajectory": str(trajectory_path),
        "validation_pairs": len(base_labels),
        "observed_probe_steps": steps,
        "step0_reference_mean_validation": first["validation"],
        "events": summary_events(rows),
        "observations": rows,
        "limitations": [
            "Only sparse saved E0 probes were analyzed; cannot identify "
            "which unobserved exact optimizer step caused a change.",
            "No complete TRAIN-set trajectory exists in these saved files; "
            "last_online_minibatch_train_mse is not comparable to full "
            "validation-set MSE or full TRAIN-set MSE.",
            "Mean+current-head MSE separates the affine-head-only "
            "counterfactual from the B2 pool representation at EACH "
            "recorded step, but is not a full multi-step causal ablation.",
            "The selected trained Mean checkpoint used this validation "
            "split for Spearman selection. This is not independent test data.",
            "Rank changes are measured relative to step-0 predictions, "
            "not directly to ground-truth ranks.",
            "STSB labels may be tied; rankdata uses average ranks.",
            "There are no saved training-set pair predictions to infer "
            "how train Spearman/MSE evolved at every probe.",
        ],
    }


def graph_bytes(report, metric):
    # Import only in full mode so --check-only never depends on the plotting
    # backend or writes plotting caches in the experiment result directory.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observations = report["observations"]
    xs = [r["step"] for r in observations]
    full = [r["validation"][metric] for r in observations]
    meanhead = [
        r["mean_pool_with_current_head_validation"][metric]
        for r in observations
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(xs, full, marker="o", label="B2 (full)")
    ax.plot(xs, meanhead, marker="s", linestyle="--",
            label="Mean pooling + current B2 head")
    ax.axhline(
        observations[0]["validation"][metric],
        linestyle=":", linewidth=1.1, label="Step-0 trained Mean")
    ax.set_xlabel("E0 optimizer step (only saved probes)")
    ax.set_ylabel(
        "Validation MSE (lower better)" if metric == "mse"
        else "Validation Spearman (higher better)")
    ax.set_title(
        "E3a / STSB seed " + str(report["seed"]) +
        " / Validation " + metric.upper())
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(s) for s in xs])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=170)
    plt.close(fig)
    return buffer.getvalue()


def write_results(root, seed, report):
    dest = root / f"seed_{seed}"
    files = {
        "json": dest / "e3_trajectory_analysis.json",
        "mse_chart": dest / "e3_val_mse.png",
        "spearman_chart": dest / "e3_val_spearman.png",
    }
    existing = [str(p) for p in files.values() if p.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing E3a result files: " +
            ", ".join(existing))
    # Prepare the entire report before touching the output directory.
    payload = (json.dumps(
        report, ensure_ascii=False, indent=2,
        allow_nan=False) + "\n").encode("utf-8")
    plot_mse = graph_bytes(report, "mse")
    plot_sp = graph_bytes(report, "spearman")
    dest.mkdir(parents=True, exist_ok=True)
    # Exclusive 'xb' rejects an existing destination, including races.
    for key, data in (("mse_chart", plot_mse),
                      ("spearman_chart", plot_sp), ("json", payload)):
        with files[key].open("xb") as stream:
            stream.write(data)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pilot_dir", default="outputs/text/stsb_frozen_trained_mean")
    parser.add_argument(
        "--output_root",
        default="outputs/text/stsb_mean_equiv_e3_trajectory")
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate and summarize E0 data without writing any output.")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative.")
    pilot = Path(args.pilot_dir) / f"seed_{args.seed}"
    print("E3a: saved E0 validation trajectory ONLY; no model/training/GPU.")
    report = analyze(pilot, args.seed)
    print(f"seed={args.seed}, validation_pairs={report['validation_pairs']}")
    print("Only saved E0 probe steps:", report["observed_probe_steps"])
    print(f"  {'step':>6} {'val_mse':>12} {'Δmse_vs_0':>12} "
          f"{'val_sp':>12} {'Δsp_vs_0':>12} {'mean_cos_Δ':>13}")
    for row in report["observations"]:
        print(f"  {row['step']:>6d} {row['validation']['mse']:>12.7f} "
              f"{row['delta_vs_step0']['mse']:>+12.7f} "
              f"{row['validation']['spearman']:>12.6f} "
              f"{row['delta_vs_step0']['spearman']:>+12.6f} "
              f"{row['mean_absolute_pair_cos_change_from_step0']:>13.6f}")
    print("Earliest OBSERVED val MSE > Step0:",
          report["events"]["first_observed_validation_mse_above_step0"])
    print("Earliest OBSERVED val Spearman < Step0:",
          report["events"]["first_observed_validation_spearman_below_step0"])
    print("Best observed MSE step:",
          report["events"]["best_observed_validation_mse_step"])
    print("Best observed Spearman step:",
          report["events"]["best_observed_validation_spearman_step"])
    if args.check_only:
        print("CHECK-ONLY: no E3a output files or folders created.")
        return
    files = write_results(Path(args.output_root), args.seed, report)
    for name, path in files.items():
        print(f"Saved fixed {name}: {path}")


if __name__ == "__main__":
    main()
