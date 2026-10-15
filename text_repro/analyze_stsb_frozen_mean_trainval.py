#!/usr/bin/env python3
"""Post-hoc, READ-ONLY evaluation of the trained-Mean -> frozen FLaG_B2 STSB pilot.

No training, no test split, and no new model selection.

Compare full-training and validation MSE for the same frozen RoBERTa:
  * Mean with original (pre-FLaG) score head
  * Mean with the LAST FLaG-trained score head
  * FLaG_B2 with its LAST trained score head
  * Mean and FLaG_B2 with SEPARATELY optimal positive-affine cosine
    score heads fitted ONLY on the training split.

Spearman is included, but does not depend on a positive affine head.
The optimal-head comparison isolates representational fitting capacity from
simple score-head calibration; it is NOT retraining a pooling model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from explore_mean_equiv_trajectory import evaluate, get_splits
from train_author_text_protocol import PairCollator, PairDataset
from train_stsb_frozen_mean_start import build_frozen_b2


def make_eval_loader(dataset, tokenizer):
    collator = PairCollator(tokenizer, max_length=128, label_dtype=torch.float32)
    return DataLoader(
        PairDataset(dataset, "stsb"),
        batch_size=64, shuffle=False, collate_fn=collator, num_workers=0,
    )


def head_scores(cosines, log_scale, bias):
    return float(np.exp(log_scale)) * cosines + float(bias)


def fit_positive_affine(train_cosine, train_labels):
    """Minimize training MSE of exp(log_scale)*cos + bias analytically.

    Constrained slope is >0, matching the actual task head; intercept free.
    Use the continuous boundary limit for negative/uninformative slope.
    """
    x = np.asarray(train_cosine, dtype=np.float64)
    y = np.asarray(train_labels, dtype=np.float64) / 5.0
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.dot(xc, xc))
    slope_unconstrained = float(np.dot(xc, yc) / denom) if denom > 1e-18 else 0.0
    scale = max(slope_unconstrained, 1e-8)
    bias = float(y.mean() - scale * x.mean())
    return {
        "scale": scale,
        "bias": bias,
        "unconstrained_scale": slope_unconstrained,
        "positive_constraint_active": slope_unconstrained <= 0.0,
    }


def score_metrics(cosine, labels, scale, bias):
    cosine = np.asarray(cosine, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    target = labels / 5.0
    scores = scale * cosine + bias
    return {
        "mse": float(np.mean((scores - target) ** 2)),
        "spearman": float(spearmanr(scores, labels).statistic),
        "pearson": float(pearsonr(scores, labels).statistic),
    }


def build_metrics(train_pairs, val_pairs, mean_head, b2_head):
    train_y = train_pairs["labels"]
    val_y = val_pairs["labels"]
    mean_train = train_pairs["cos_same_encoder_mean"]
    mean_val = val_pairs["cos_same_encoder_mean"]
    flag_train = train_pairs["cos_full"]
    flag_val = val_pairs["cos_full"]

    mean_fit = fit_positive_affine(mean_train, train_y)
    flag_fit = fit_positive_affine(flag_train, train_y)
    configurations = {
        "mean_original_head": {
            "train_cos": mean_train, "val_cos": mean_val,
            **mean_head,
        },
        "mean_with_last_b2_head": {
            "train_cos": mean_train, "val_cos": mean_val,
            **b2_head,
        },
        "b2_with_last_b2_head": {
            "train_cos": flag_train, "val_cos": flag_val,
            **b2_head,
        },
        "mean_best_affine_fit_on_train": {
            "train_cos": mean_train, "val_cos": mean_val,
            "scale": mean_fit["scale"], "bias": mean_fit["bias"],
        },
        "b2_best_affine_fit_on_train": {
            "train_cos": flag_train, "val_cos": flag_val,
            "scale": flag_fit["scale"], "bias": flag_fit["bias"],
        },
    }
    result = {}
    for key, v in configurations.items():
        scale, bias = v["scale"], v["bias"]
        result[key] = {
            "head": {"scale": scale, "bias": bias},
            "train": score_metrics(v["train_cos"], train_y, scale, bias),
            "validation": score_metrics(v["val_cos"], val_y, scale, bias),
        }
    result["calibration_info"] = {
        "mean_fit": mean_fit,
        "b2_fit": flag_fit,
        "note": "Best affine heads optimized on TRAIN labels only. "
                "Validation labels are never used to fit the head.",
    }
    return result


def print_table(results):
    print("\nSTSB | TRAIN and VALIDATION | identical frozen RoBERTa")
    print("All MSE values use normalized gold label = STSB score / 5.")
    print(f"{'Pooling + score head':<34} {'Train MSE':>11}"
          f" {'Val MSE':>11} {'Train SP':>11} {'Val SP':>11}")
    print("-" * 84)
    for key, value in results.items():
        if key == "calibration_info":
            continue
        t, v = value["train"], value["validation"]
        print(f"{key:<34} {t['mse']:>11.6f} {v['mse']:>11.6f}"
              f" {t['spearman']:>11.6f} {v['spearman']:>11.6f}")
    b = results["b2_with_last_b2_head"]
    m = results["mean_with_last_b2_head"]
    bm = results["b2_best_affine_fit_on_train"]
    mm = results["mean_best_affine_fit_on_train"]
    print("\nKey DIFFERENCES (FLaG_B2 minus Mean; negative MSE is better):")
    for phase in ("train", "validation"):
        print(f"  {phase}: fixed-last-head MSE = "
              f"{b[phase]['mse']-m[phase]['mse']:+.6f}, "
              f"optimally-calibrated MSE = "
              f"{bm[phase]['mse']-mm[phase]['mse']:+.6f}, "
              f"Spearman = {b[phase]['spearman']-m[phase]['spearman']:+.6f}")
    print("\nINTERPRETATION:")
    train_gap = bm["train"]["mse"] - mm["train"]["mse"]
    val_gap = bm["validation"]["mse"] - mm["validation"]["mse"]
    if train_gap < -1e-6 and val_gap > 1e-6:
        print("  Extra B2 representation fits training MSE better even after "
              "head calibration, but has worse validation MSE.")
        print("  This supports a train/validation gap worth investigating; "
              "it does not by itself prove overfitting as the sole cause.")
    elif train_gap >= -1e-6:
        print("  Extra B2 representation shows no clear training MSE gain "
              "over Mean once both heads are fitted optimally.")
        print("  Check the learning trajectory or optimization before "
              "attributing the validation loss to overfitting.")
    else:
        print("  Extra B2 representation improves training MSE, while "
              "the validation result requires closer comparison.")
    print("  No causal claim about Gate, Projection, or gradient behavior "
          "can be inferred from these metrics alone.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_path", default="/home/data/home/wwr_lumos/models/roberta-base")
    p.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    p.add_argument("--pilot_dir", default="outputs/text/stsb_frozen_trained_mean")
    p.add_argument("--output_file", default="")
    args = p.parse_args()

    pilot = Path(args.pilot_dir) / f"seed_{args.seed}"
    summary_path = pilot / "summary.json"
    last_path = pilot / "last_pool_head.pt"
    if not summary_path.is_file() or not last_path.is_file():
        raise FileNotFoundError(
            "Expected prior completed frozen-Mean pilot under "
            f"{pilot}. Do NOT retrain just for this analysis."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("frozen_roberta") is not True:
        raise ValueError("Expected a frozen-RoBERTa pilot")
    if summary.get("seed") != args.seed:
        raise ValueError("Seed mismatch in summary.json")

    last = torch.load(last_path, map_location="cpu", weights_only=True)
    checkpoint = Path(summary["mean_reference_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Trained Mean reference missing: {checkpoint}")
    if Path(last["reference_checkpoint"]) != checkpoint:
        raise ValueError(
            "Last FLaG snapshot refers to a DIFFERENT Mean checkpoint. "
            "Refusing to compare unmatched encoders."
        )
    ref_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(k.startswith("pool.") for k in ref_state):
        raise ValueError("Reference state dict does not appear to be Mean")
    if int(last["step"]) != int(summary["steps"]):
        raise ValueError("Last checkpoint step does not match experiment summary")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True,
    )
    train, val = get_splits(
        "stsb", args.stsb_data_path, "", args.seed, 0, 0,
    )
    if (len(train) != summary["n_train"]
            or len(val) != summary["n_validation"]):
        raise ValueError("STSB split sizes no longer match original experiment")

    model = build_frozen_b2(args, device, ref_state)
    model.pool.load_state_dict(last["pool"], strict=True)
    with torch.no_grad():
        model.log_scale.copy_(last["log_scale"].to(device))
        model.bias.copy_(last["bias"].to(device))
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("RoBERTa must be frozen for this comparison")

    print("[Post-hoc analysis] No optimizer, no backward, no model training.",
          flush=True)
    print(f"device={device}, seed={args.seed}", flush=True)
    print("Evaluating all 5749 STSB TRAIN pairs...", flush=True)
    _, train_pairs = evaluate(
        model, make_eval_loader(train, tokenizer), "stsb", device,
    )
    print("Evaluating all 1500 STSB VALIDATION pairs...", flush=True)
    _, val_pairs = evaluate(
        model, make_eval_loader(val, tokenizer), "stsb", device,
    )

    # Regression check: the reloaded LAST B2 model must match the previous
    # output on the same held-out validation set before interpreting results.
    b2_val_head = {
        "scale": float(torch.exp(last["log_scale"]).item()),
        "bias": float(last["bias"].item()),
    }
    orig_head = {
        "scale": float(torch.exp(ref_state["log_scale"]).item()),
        "bias": float(ref_state["bias"].item()),
    }
    results = build_metrics(train_pairs, val_pairs, orig_head, b2_val_head)
    expected_b2 = float(summary["final_b2_val_spearman"])
    obtained_b2 = results["b2_with_last_b2_head"]["validation"]["spearman"]
    expected_ref = float(summary["mean_reference_validation"]["spearman"])
    obtained_ref = results["mean_original_head"]["validation"]["spearman"]
    if (abs(obtained_b2 - expected_b2) > 5e-4
            or abs(obtained_ref - expected_ref) > 5e-4):
        raise RuntimeError(
            "Reloaded checkpoint does not reproduce earlier validation "
            "Spearman. Check model/data versions and paths: "
            f"B2 {obtained_b2:.6f} vs {expected_b2:.6f}; "
            f"Mean {obtained_ref:.6f} vs {expected_ref:.6f}."
        )
    b2_expected_mse = float(
        summary.get("mean_reference_validation", {}).get("mse", float("nan"))
    )
    if not np.isfinite(b2_expected_mse):
        raise RuntimeError("Reference MSE missing in summary.json")

    output_path = (
        Path(args.output_file) if args.output_file
        else pilot / "posthoc_train_val_comparison.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "Post-hoc train/validation pool-vs-Mean comparison",
        "seed": args.seed,
        "reference_checkpoint": str(checkpoint),
        "b2_checkpoint": str(last_path),
        "b2_step": int(last["step"]),
        "frozen_backbone": True,
        "train_pairs": len(train),
        "validation_pairs": len(val),
        "test_used": False,
        "training_performed": False,
        "heads_fit_with": "Training set only (least squares, positive scale)",
        "metrics": results,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_table(results)
    print(f"\nSaved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
