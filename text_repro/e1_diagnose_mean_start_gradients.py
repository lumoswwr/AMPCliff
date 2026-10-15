#!/usr/bin/env python3
"""E1: read-only initial-gradient diagnosis at the trained-Mean-equivalent B2 point.

No optimizer is constructed, no optimizer.step/backward updates are performed, and
no reference checkpoint or previous experiment output is overwritten.

Scientific question:
  Is the full-training-set gradient nonzero in the extra B2 directions?
  Do train/validation MSE gradients agree, and how variable are minibatch
  gradients relative to the full training gradient?

Run from AMPCliff root with PYTHONPATH containing the repository parent and
text_repro (see text_repro/E1_MEAN_EQUIV_GRADIENTS.md).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from explore_mean_equiv_trajectory import evaluate, get_loaders, get_splits
from train_stsb_frozen_mean_start import (
    assert_initial_pool_identity,
    build_frozen_b2,
)


GROUPS = ("gate_output", "projection", "head")
ZERO_GRAD_TOLERANCE = 1e-9


def parameter_groups(model):
    """Identify paths which can have nonzero first-step gradients at B2 init."""
    gate_ids = {id(p) for p in model.pool.freq_gate[-1].parameters()}
    projection_ids = {id(p) for p in model.pool.time_out_proj.parameters()}
    head_ids = {id(model.log_scale), id(model.bias)}
    named_all = [(name, p) for name, p in model.named_parameters()
                 if p.requires_grad]
    chosen_ids = gate_ids | projection_ids | head_ids
    named_active = [(name, p) for name, p in named_all
                    if id(p) in chosen_ids]
    groups = {}
    for name, p in named_active:
        pid = id(p)
        groups[name] = ("gate_output" if pid in gate_ids
                        else "projection" if pid in projection_ids
                        else "head")
    if set(groups.values()) != set(GROUPS):
        raise RuntimeError("Missing one of gate_output/projection/head.")
    if len(named_active) != 6 or len(chosen_ids) != 6:
        raise RuntimeError("Expected exactly 6 active tensors (2 per group).")
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("RoBERTa encoder must be frozen.")
    if not set(chosen_ids).issubset({id(p) for _, p in named_all}):
        raise RuntimeError("Expected active parameters not found.")
    return named_all, named_active, groups


def tensor_norm(grads):
    return math.sqrt(sum(float(torch.sum(v * v)) for v in grads))


def vector_stats(left, right):
    """Cosine and dot product, both based on CPU float64 accumulation."""
    dot = sum(float(torch.sum(x * y)) for x, y in zip(left, right))
    lnorm, rnorm = tensor_norm(left), tensor_norm(right)
    cos = dot / (lnorm * rnorm) if lnorm > 1e-12 and rnorm > 1e-12 else None
    if cos is not None:
        cos = float(max(-1.0, min(1.0, cos)))
    return {
        "train_norm": lnorm,
        "validation_norm": rnorm,
        "dot_train_val": dot,
        "cos_train_val": cos,
        # For a sufficiently small plain SGD step -alpha*g_train:
        # delta(val MSE) = alpha * this coefficient + O(alpha^2).
        # This is NOT the actual first Adam step and NOT a Spearman estimate.
        "sgd_first_order_validation_mse_coefficient": -dot,
    }


def group_slices(named_active, group_of):
    return {group: [i for i, (name, _) in enumerate(named_active)
                    if group_of[name] == group]
            for group in (*GROUPS, "all")}


def subset(grads, indices):
    return [grads[i] for i in indices]


def run_gradient_pass(model, loader, total_pairs, device, named_all,
                      named_active, sample_positions=None):
    """Compute the dataset-average MSE gradient in a single streaming pass.

    Individual minibatch losses have mean reduction. Aggregate their gradients
    with weight batch_size / total_pairs, NOT 1 / num_batches, so the final
    shorter minibatch has the correct weight.

    Cached minibatch gradients are deterministic, disjoint minibatches from the
    same shuffled loader pass, preserving actual dynamic-padding behavior.
    """
    model.eval()
    all_params = [p for _, p in named_all]
    active_params = [p for _, p in named_active]
    active_indices = [next(i for i, (_, p) in enumerate(named_all)
                           if id(p) == id(target))
                      for _, target in named_active]
    sums = [torch.zeros_like(p, dtype=torch.float64, device="cpu")
            for p in active_params]
    snapshots = []
    selected = set(sample_positions or [])
    n_seen = 0
    weighted_loss = 0.0
    upstream_sanity = None

    for batch_index, (tok_a, tok_b, labels) in enumerate(loader):
        tok_a = {k: v.to(device) for k, v in tok_a.items()}
        tok_b = {k: v.to(device) for k, v in tok_b.items()}
        target = labels.to(device) / 5.0
        predicted, _ = model(tok_a, tok_b)
        loss = F.mse_loss(predicted, target)
        n = int(labels.numel())

        if batch_index == 0:
            # Do this once to verify the theoretical zero-initialization
            # barrier for the gate's upstream/latent parameters.
            all_grads = torch.autograd.grad(
                loss, all_params, allow_unused=True)
            active_grads = [all_grads[i] for i in active_indices]
            active_id_set = {id(p) for p in active_params}
            upstream_sanity = {}
            for (name, p), grad in zip(named_all, all_grads):
                if id(p) in active_id_set:
                    continue
                max_abs = (0.0 if grad is None
                           else float(grad.detach().abs().max().item()))
                upstream_sanity[name] = {
                    "gradient_is_none": grad is None,
                    "max_abs_gradient": max_abs,
                }
                if max_abs > ZERO_GRAD_TOLERANCE:
                    raise RuntimeError(
                        "Nonzero upstream initialization gradient for "
                        f"{name}: {max_abs:.3e}; abort and inspect B2 wiring.")
        else:
            active_grads = torch.autograd.grad(
                loss, active_params, allow_unused=False)

        for i, grad in enumerate(active_grads):
            if grad is None:
                raise RuntimeError(
                    f"Missing active gradient: {named_active[i][0]}")
            if not bool(torch.isfinite(grad).all().item()):
                raise RuntimeError(
                    f"Nonfinite gradient: {named_active[i][0]}")
            cpu_grad = grad.detach().to(device="cpu", dtype=torch.float64)
            sums[i].add_(cpu_grad, alpha=n / total_pairs)

        if batch_index in selected:
            snapshots.append({
                "batch_index": batch_index,
                "batch_pairs": n,
                "grads": [g.detach().to("cpu", dtype=torch.float32).clone()
                          for g in active_grads],
            })
        weighted_loss += float(loss.item()) * n
        n_seen += n
        if (batch_index + 1) % 50 == 0:
            print(f"  gradient batches {batch_index + 1}/{len(loader)} "
                  f"({n_seen}/{total_pairs} pairs)", flush=True)

    if n_seen != total_pairs:
        raise RuntimeError(
            f"Dataset count mismatch: {n_seen} vs {total_pairs}")
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Unexpected .grad accumulation; expected autograd.grad.")
    return {
        "gradient": sums,
        "mse": weighted_loss / total_pairs,
        "sampled_minibatches": snapshots,
        "upstream_zero_barrier": upstream_sanity,
        "batches": len(loader),
        "pairs": n_seen,
    }


def summarize_minibatches(sampled, full_grads, slices):
    result = {}
    for group, idx in slices.items():
        reference = subset(full_grads, idx)
        ref_norm = tensor_norm(reference)
        rows, rms_noise_terms = [], []
        for sample in sampled:
            g = [sample["grads"][i].to(dtype=torch.float64) for i in idx]
            dot = sum(float(torch.sum(a * b))
                      for a, b in zip(g, reference))
            norm = tensor_norm(g)
            cos = (dot / (norm * ref_norm)
                   if norm > 1e-12 and ref_norm > 1e-12 else None)
            noise_sq = sum(float(torch.sum((a - b) ** 2))
                           for a, b in zip(g, reference))
            rms_noise_terms.append(noise_sq)
            rows.append({
                "batch_index_zero_based": sample["batch_index"],
                "batch_pairs": sample["batch_pairs"],
                "gradient_norm": norm,
                "cosine_with_full_train_gradient": (
                    max(-1., min(1., cos)) if cos is not None else None),
            })
        valid = [r["cosine_with_full_train_gradient"] for r in rows
                 if r["cosine_with_full_train_gradient"] is not None]
        result[group] = {
            "full_train_gradient_norm": ref_norm,
            "sampled_batch_count": len(rows),
            "mean_minibatch_cosine_with_full_train": (
                statistics.mean(valid) if valid else None),
            "fraction_negative_cosine": (
                sum(x < 0 for x in valid) / len(valid) if valid else None),
            # Descriptive RMS for the selected minibatches, NOT a confidence
            # interval or proof that stochasticity causes the final outcome.
            "sampled_rms_deviation_over_full_gradient_norm": (
                math.sqrt(statistics.mean(rms_noise_terms)) / ref_norm
                if rms_noise_terms and ref_norm > 1e-12 else None),
            "minibatches": rows,
        }
    return result


def save_unique_result(root, seed, report):
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(tempfile.mkdtemp(prefix=f"run_{timestamp}_", dir=seed_dir))
    output = out_dir / "e1_gradient_diagnostic.json"
    # Exclusive creation is a second no-clobber safeguard.
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_path", default=(
        "/home/data/home/wwr_lumos/models/roberta-base"))
    parser.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    parser.add_argument("--pilot_dir",
                        default="outputs/text/stsb_frozen_trained_mean")
    parser.add_argument("--output_root",
                        default="outputs/text/stsb_mean_equiv_e1_gradients")
    parser.add_argument("--sample_minibatches", type=int, default=16)
    parser.add_argument("--train_limit", type=int, default=0,
                        help="Smoke test ONLY; 0 uses all training pairs.")
    parser.add_argument("--val_limit", type=int, default=0,
                        help="Smoke test ONLY; 0 uses all validation pairs.")
    args = parser.parse_args()

    if args.seed < 0 or args.sample_minibatches < 1:
        parser.error("seed >= 0 and sample_minibatches >= 1 required.")
    if args.train_limit < 0 or args.val_limit < 0:
        parser.error("limits must be nonnegative.")
    is_full = args.train_limit == 0 and args.val_limit == 0

    pilot = Path(args.pilot_dir) / f"seed_{args.seed}"
    summary_file = pilot / "summary.json"
    if not summary_file.is_file():
        raise FileNotFoundError(f"Existing E0 summary required: {summary_file}")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if summary.get("seed") != args.seed or summary.get("frozen_roberta") is not True:
        raise RuntimeError("E0 summary seed/frozen-backbone mismatch.")

    refpath = Path(summary["mean_reference_checkpoint"])
    if not refpath.is_file():
        raise FileNotFoundError(f"Trained Mean checkpoint missing: {refpath}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("E1 read-only trained-Mean-equivalent B2 gradient diagnosis")
    print(f"seed={args.seed} device={device} reference={refpath}")
    print("NO optimizer; NO parameter updates; NO existing outputs overwritten.")
    if not is_full:
        print("WARNING: subset/smoke mode; NOT a full-dataset scientific result.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True)
    train_data, val_data = get_splits(
        "stsb", args.stsb_data_path, "", args.seed,
        args.train_limit, args.val_limit)
    if is_full and (len(train_data) != summary["n_train"]
                    or len(val_data) != summary["n_validation"]):
        raise RuntimeError("STSB data sizes differ from original E0 run.")
    train_loader, val_loader = get_loaders(
        "stsb", tokenizer, train_data, val_data, args.seed)

    ref_state = torch.load(refpath, map_location="cpu", weights_only=True)
    if not any(k.startswith("encoder.") for k in ref_state):
        raise RuntimeError("Missing Mean reference encoder weights.")
    if any(k.startswith("pool.") for k in ref_state):
        raise RuntimeError("Reference checkpoint is not a Mean model.")
    model = build_frozen_b2(args, device, ref_state)
    del ref_state
    model.eval()
    named_all, named_active, group_of = parameter_groups(model)
    slices = group_slices(named_active, group_of)
    slices["all"] = list(range(len(named_active)))

    initial_weights = {name: p.detach().cpu().clone()
                       for name, p in model.named_parameters()
                       if p.requires_grad}
    pool_max_diff = assert_initial_pool_identity(model, val_loader, device)
    if pool_max_diff > 2e-5:
        raise RuntimeError("Mean-equivalent initialization FAILED.")

    print("Checking B2 vs same-encoder Mean predictions on validation...")
    val_metrics, val_arrays = evaluate(model, val_loader, "stsb", device)
    score_max_diff = float(np.max(np.abs(
        val_arrays["score_full"] - val_arrays["score_same_encoder_mean"])))
    if score_max_diff > 2e-4:
        raise RuntimeError(
            f"Initial B2/Mean prediction identity failed: {score_max_diff}")
    if is_full:
        expected_sp = float(summary["mean_reference_validation"]["spearman"])
        observed_sp = val_metrics["same_encoder_mean"]["spearman"]
        if abs(expected_sp - observed_sp) > 5e-4:
            raise RuntimeError(
                f"Reference val Spearman changed: E0={expected_sp:.6f}, "
                f"E1={observed_sp:.6f}. Inspect data/model first.")

    sampled_positions = sorted(set(np.linspace(
        0, len(train_loader) - 1,
        num=min(args.sample_minibatches, len(train_loader)),
        dtype=int).tolist()))
    print(f"Computing TRAIN dataset gradient: {len(train_data)} pairs; "
          f"{len(train_loader)} batches...")
    train = run_gradient_pass(
        model, train_loader, len(train_data), device, named_all, named_active,
        sampled_positions)
    print(f"Computing VALIDATION dataset gradient: {len(val_data)} pairs; "
          f"{len(val_loader)} batches...")
    validation = run_gradient_pass(
        model, val_loader, len(val_data), device, named_all, named_active)
    if abs(validation["mse"] - val_metrics["full"]["mse"]) > 2e-5:
        raise RuntimeError(
            "Validation MSE disagreement between eval and gradient pass; "
            "check padding, model state and loss normalization.")

    # Exact parameter value check after all forward/autograd operations.
    unchanged = all(torch.equal(p.detach().cpu(), initial_weights[name])
                    for name, p in model.named_parameters()
                    if p.requires_grad)
    if not unchanged:
        raise RuntimeError("A trainable parameter changed during E1; abort.")

    alignment = {
        name: vector_stats(subset(train["gradient"], idx),
                           subset(validation["gradient"], idx))
        for name, idx in slices.items()
    }
    minibatch = summarize_minibatches(
        train["sampled_minibatches"], train["gradient"], slices)
    report = {
        "experiment": "E1_mean_equivalent_initial_gradient",
        "research_protocol": "frozen trained Mean -> strict B2 identity, step 0",
        "seed": args.seed,
        "full_dataset_result": is_full,
        "reference_checkpoint": str(refpath),
        "reference_summary": str(summary_file),
        "model_path": args.model_path,
        "data_path": args.stsb_data_path,
        "device": str(device),
        "torch_version": torch.__version__,
        "train_batch_size": 32,
        "validation_batch_size": 64,
        "dynamic_padding": True,
        "sampled_train_minibatches": sampled_positions,
        "loss": "MSE(score, STSB_label/5)",
        "encoder_frozen": True,
        "optimizer_constructed": False,
        "parameter_updates": 0,
        "trainable_parameters_unchanged_exactly": unchanged,
        "train_pairs": train["pairs"],
        "val_pairs": validation["pairs"],
        "train_mse": train["mse"],
        "val_mse": validation["mse"],
        "initial_identity": {
            "max_abs_pool_output_difference": pool_max_diff,
            "max_abs_pair_score_difference": score_max_diff,
            "validation_spearman": val_metrics["full"]["spearman"],
        },
        "active_parameters": [
            {"name": name, "group": group_of[name],
             "numel": p.numel()}
            for name, p in named_active
        ],
        "initial_upstream_gradient_sanity_first_train_batch":
            train["upstream_zero_barrier"],
        "train_vs_validation_gradients": alignment,
        "minibatch_gradients": minibatch,
        "limitations": [
            "No optimizer or parameter updates; this is not a causal "
            "measurement of actual Adam update effects.",
            "Train/val gradient cosine is for MSE, not Spearman.",
            "The existing validation split selected the reference Mean "
            "checkpoint; it is not an untouched test set.",
            "Each full gradient uses the exact deterministic loader batches "
            "and their dynamic padding; the objective is batch-shape-dependent "
            "for FLaG in general.",
            "Selected minibatches are descriptive rather than an independent "
            "estimate of a general population gradient covariance.",
        ],
    }
    output = save_unique_result(Path(args.output_root), args.seed, report)

    print("\nE1 initial-gradient summary (MSE):")
    print(f"  initial pool max difference:  {pool_max_diff:.3e}")
    print(f"  initial score max difference: {score_max_diff:.3e}")
    print(f"  training MSE:                 {train['mse']:.8f}")
    print(f"  validation MSE:               {validation['mse']:.8f}")
    print(f"  {'group':<16} {'train_norm':>13} {'val_norm':>13} "
          f"{'cosine':>11} {'mini_rms/full':>15}")
    for group in (*GROUPS, "all"):
        a, m = alignment[group], minibatch[group]
        cosine = a["cos_train_val"]
        noise = m["sampled_rms_deviation_over_full_gradient_norm"]
        print(f"  {group:<16} {a['train_norm']:>13.5e} "
              f"{a['validation_norm']:>13.5e} "
              f"{cosine if cosine is not None else float('nan'):>11.5f} "
              f"{noise if noise is not None else float('nan'):>15.3f}")
    print(f"  unchanged weights: {unchanged}")
    print(f"\nSaved NEW result: {output}", flush=True)


if __name__ == "__main__":
    main()
