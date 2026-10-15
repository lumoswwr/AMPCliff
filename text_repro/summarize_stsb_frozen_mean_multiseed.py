#!/usr/bin/env python3
"""Summarize pre-existing frozen trained-Mean -> FLaG_B2 STSB pilot across seeds.

READ ONLY over existing run files; writes aggregate CSV/JSON to the output root.
Primary endpoint pre-specified: LAST trained B2 vs SAME seed's trained Mean
reference on STSB validation Spearman. No test metrics, no new training.
Secondary endpoint: posthoc TRAIN/VAL MSE with separate positive-affine heads
fitted on TRAIN only. All requested seeds must have complete paired results.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def finite_number(value, name):
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"Nonfinite {name}: {value!r}")
    return v


def close(a, b, label, tol=5e-4):
    if abs(a - b) > tol:
        raise ValueError(f"{label}: {a:.8f} != {b:.8f} (tol={tol})")


def read_one(root, seed):
    run = root / f"seed_{seed}"
    summary = read_json(run / "summary.json")
    posthoc = read_json(run / "posthoc_train_val_comparison.json")
    history_path = run / "trajectory.jsonl"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing trajectory: {history_path}")
    with history_path.open(encoding="utf-8") as handle:
        trajectory = [json.loads(line) for line in handle if line.strip()]
    if not trajectory:
        raise ValueError(f"Empty trajectory for seed {seed}")

    if summary["seed"] != seed or posthoc["seed"] != seed:
        raise ValueError(f"Seed mismatch in {run}")
    if summary.get("frozen_roberta") is not True or posthoc.get("frozen_backbone") is not True:
        raise ValueError(f"Unfrozen comparison, seed {seed}")
    if posthoc.get("training_performed") is not False or posthoc.get("test_used") is not False:
        raise ValueError(f"Posthoc includes training/test use, seed {seed}")
    if summary.get("dataset") != "STSB official train and validation; test not evaluated":
        raise ValueError(f"Unexpected dataset protocol, seed {seed}")
    if summary.get("flag_pool") != "masked_mean" or summary.get("post_pool_norm") is not False:
        raise ValueError(f"Not mean-equivalent B2 protocol, seed {seed}")
    if summary.get("flag_epochs") != 3:
        raise ValueError(f"Expected exactly 3 frozen B2 epochs, seed {seed}")
    if (int(summary["n_train"]) != int(posthoc["train_pairs"])
            or int(summary["n_validation"]) != int(posthoc["validation_pairs"])):
        raise ValueError(f"Dataset count mismatch, seed {seed}")
    if int(summary["n_train"]) != 5749 or int(summary["n_validation"]) != 1500:
        raise ValueError(f"Unexpected STSB split sizes, seed {seed}")
    if int(summary["steps"]) != int(posthoc["b2_step"]):
        raise ValueError(f"Posthoc checkpoint step mismatch, seed {seed}")
    if posthoc["reference_checkpoint"] != summary["mean_reference_checkpoint"]:
        raise ValueError(f"Reference Mean mismatch, seed {seed}")
    if trajectory[0].get("global_step") != 0:
        raise ValueError(f"Missing B2==Mean initial step, seed {seed}")
    if trajectory[-1].get("global_step") != summary["steps"]:
        raise ValueError(f"Last trajectory step mismatch, seed {seed}")

    measured = posthoc["metrics"]
    mn = measured["mean_best_affine_fit_on_train"]
    fl = measured["b2_best_affine_fit_on_train"]
    mn_raw = measured["mean_original_head"]
    fl_final_head = measured["b2_with_last_b2_head"]
    mn_final_head = measured["mean_with_last_b2_head"]
    reference = finite_number(summary["mean_reference_validation"]["spearman"], "mean Spearman")
    b2_final = finite_number(summary["final_b2_val_spearman"], "B2 final Spearman")
    close(reference, float(mn_raw["validation"]["spearman"]), "Mean posthoc Spearman")
    close(b2_final, float(fl_final_head["validation"]["spearman"]), "B2 posthoc Spearman")
    close(b2_final - reference, float(summary["final_b2_minus_mean_spearman"]),
          "B2 final minus Mean")
    close(float(trajectory[0]["full_flag_b2"]["spearman"]), reference,
          "B2 init vs strong Mean", tol=5e-6)

    mean_train_sp = finite_number(mn["train"]["spearman"], "Mean train Spearman")
    b2_train_sp = finite_number(fl["train"]["spearman"], "B2 train Spearman")
    mean_val_sp = finite_number(mn["validation"]["spearman"], "Mean val Spearman")
    b2_val_sp = finite_number(fl["validation"]["spearman"], "B2 val Spearman")
    close(reference, mean_val_sp, "Mean fitted-head validation Spearman")
    close(b2_final, b2_val_sp, "B2 fitted-head validation Spearman")

    initial_pool_error = finite_number(summary["b2_init_max_abs_pool_difference"], "pool error")
    initial_pred_error = finite_number(
        summary["b2_init_max_abs_prediction_difference"], "prediction error"
    )
    if initial_pool_error > 2e-5 or initial_pred_error > 2e-4:
        raise ValueError(f"Initial B2/Mean equivalence failed, seed {seed}")
    if len([p for p in trajectory if p["global_step"] > 0]) < 3:
        raise ValueError(f"Too few post-training probes, seed {seed}")
    positive_steps = [p for p in trajectory if p["global_step"] > 0]
    post_training_best_sp = max(
        finite_number(p["full_flag_b2"]["spearman"], "posttrain val Spearman")
        for p in positive_steps
    )
    best_posttrain_step = next(
        int(p["global_step"]) for p in positive_steps
        if abs(float(p["full_flag_b2"]["spearman"]) - post_training_best_sp) < 1e-12
    )
    mean_train_mse = finite_number(mn["train"]["mse"], "Mean train MSE")
    b2_train_mse = finite_number(fl["train"]["mse"], "B2 train MSE")
    mean_val_mse = finite_number(mn["validation"]["mse"], "Mean val MSE")
    b2_val_mse = finite_number(fl["validation"]["mse"], "B2 val MSE")
    mean_head_train = finite_number(
        mn_final_head["train"]["mse"], "Mean fixed-head train MSE")
    b2_head_train = finite_number(
        fl_final_head["train"]["mse"], "B2 fixed-head train MSE")
    mean_head_val = finite_number(
        mn_final_head["validation"]["mse"], "Mean fixed-head val MSE")
    b2_head_val = finite_number(
        fl_final_head["validation"]["mse"], "B2 fixed-head val MSE")
    return {
        "seed": seed,
        "reference_source": summary["mean_reference_origin"],
        "mean_reference_val_spearman": reference,
        "b2_final_val_spearman": b2_final,
        "final_val_spearman_delta": b2_final - reference,
        "b2_best_posttraining_val_spearman": post_training_best_sp,
        "b2_best_posttraining_step": best_posttrain_step,
        "best_posttraining_val_spearman_delta": post_training_best_sp - reference,
        "b2_best_overall_step_including_initial": summary["best_b2_step"],
        "mean_train_spearman": mean_train_sp,
        "b2_train_spearman": b2_train_sp,
        "train_spearman_delta": b2_train_sp - mean_train_sp,
        "mean_train_mse_fit_head": mean_train_mse,
        "b2_train_mse_fit_head": b2_train_mse,
        "train_mse_delta_fit_head": b2_train_mse - mean_train_mse,
        "mean_val_mse_fit_head": mean_val_mse,
        "b2_val_mse_fit_head": b2_val_mse,
        "val_mse_delta_fit_head": b2_val_mse - mean_val_mse,
        "train_mse_delta_shared_last_head": b2_head_train - mean_head_train,
        "val_mse_delta_shared_last_head": b2_head_val - mean_head_val,
        "mean_absolute_pair_cos_change_end": finite_number(
            trajectory[-1]["mean_abs_pair_cos_change"], "pair cosine change"
        ),
        "init_max_abs_pool_error": initial_pool_error,
        "init_max_abs_prediction_error": initial_pred_error,
    }


def stats(values):
    values = [finite_number(x, "aggregate statistic") for x in values]
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/text/stsb_frozen_trained_mean")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = p.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or not args.seeds:
        raise ValueError("Seeds must be nonempty and unique")
    root = Path(args.root)
    # STRICT: never quietly analyze just the successful seeds. This would
    # bias the apparent frequency of a decrease or improvement.
    rows = [read_one(root, seed) for seed in args.seeds]

    numeric_keys = [
        "mean_reference_val_spearman", "b2_final_val_spearman",
        "final_val_spearman_delta", "best_posttraining_val_spearman_delta",
        "train_spearman_delta", "train_mse_delta_fit_head",
        "val_mse_delta_fit_head", "train_mse_delta_shared_last_head",
        "val_mse_delta_shared_last_head", "mean_absolute_pair_cos_change_end",
    ]
    summary = {
        "experiment": "STSB multi-seed: start at a good fine-tuned Mean, freeze "
                      "RoBERTa, train B2 pooling and head for 3 epochs",
        "seed_set": args.seeds,
        "n": len(rows),
        "primary_endpoint": "FINAL (after 3 epochs) B2 minus SAME-SEED "
                            "trained Mean validation Spearman",
        "secondary_endpoints": "Full-train/full-validation MSE with separate "
                               "positive-affine heads fitted on TRAIN only",
        "validation_checkpoint_selection_note": "Mean reference checkpoint "
            "selected by STSB validation Spearman. The same validation set "
            "evaluates B2 adaptation. This is exploratory, not untouched-test "
            "generalization estimation.",
        "best_posttraining_note": "Exploratory maximum among >0 steps "
            "(validation-selected, not a pre-specified final endpoint).",
        "all_requested_seeds_required": True,
        "counts": {
            "final_b2_val_spearman_lower_than_mean":
                sum(x["final_val_spearman_delta"] < 0 for x in rows),
            "final_b2_val_spearman_higher_than_mean":
                sum(x["final_val_spearman_delta"] > 0 for x in rows),
            "best_posttraining_b2_val_spearman_lower_than_mean":
                sum(x["best_posttraining_val_spearman_delta"] < 0 for x in rows),
            "b2_train_spearman_higher_than_mean":
                sum(x["train_spearman_delta"] > 0 for x in rows),
            "b2_train_mse_lower_than_mean_fit_head":
                sum(x["train_mse_delta_fit_head"] < 0 for x in rows),
            "b2_val_mse_higher_than_mean_fit_head":
                sum(x["val_mse_delta_fit_head"] > 0 for x in rows),
            "train_gain_val_loss_in_spearman":
                sum(x["train_spearman_delta"] > 0 and
                    x["final_val_spearman_delta"] < 0 for x in rows),
            "train_gain_val_loss_in_calibrated_mse":
                sum(x["train_mse_delta_fit_head"] < 0 and
                    x["val_mse_delta_fit_head"] > 0 for x in rows),
        },
        "stats": {key: stats([row[key] for row in rows]) for key in numeric_keys},
        "per_seed": rows,
    }

    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "multiseed_0_4_comparison.csv"
    json_path = root / "multiseed_0_4_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("\nSTSB | frozen strong-Mean encoder | FLaG_B2 across seeds")
    print("Main metric: validation Spearman at FINAL epoch (not best checkpoint)")
    print(f"{'seed':>5} {'mean val':>11} {'B2 final':>11} {'val Δ':>10} "
          f"{'train SP Δ':>11} {'train MSE Δ':>13} {'val MSE Δ':>12}")
    for r in rows:
        print(f"{r['seed']:>5} {r['mean_reference_val_spearman']:>11.6f} "
              f"{r['b2_final_val_spearman']:>11.6f} "
              f"{r['final_val_spearman_delta']:>+10.6f} "
              f"{r['train_spearman_delta']:>+11.6f} "
              f"{r['train_mse_delta_fit_head']:>+13.6f} "
              f"{r['val_mse_delta_fit_head']:>+12.6f}")
    s = summary["stats"]["final_val_spearman_delta"]
    print(f"\nFinal Spearman Δ (B2-Mean): {s['mean']:+.6f} ± "
          f"{s['sample_std']:.6f} sample SD across {len(rows)} seeds")
    print("Counts:", summary["counts"])
    print("Important: this is an exploratory SAME-validation comparison, "
          "not published test-set performance and not causal mechanism proof.")
    print(f"Saved:\n  {csv_path}\n  {json_path}")


if __name__ == "__main__":
    main()
