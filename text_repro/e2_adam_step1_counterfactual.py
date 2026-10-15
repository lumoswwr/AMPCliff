#!/usr/bin/env python3
"""E2: exact E0 Adam step-1 reproduction and module-spliced counterfactuals.

Only one Adam update is made in memory, from the seed-matched trained Mean.
No existing checkpoint is modified. Six configurations splice parameter
values from step 0 vs the *same jointly trained* step 1, rather than train
six separate optimizers. The original E0 validation_pairs_step_1.npz must
match before the results are accepted.

Output is ONE predictable JSON path, created exclusively (no overwrite).
The optional --check-only validates the inputs but writes no output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from explore_mean_equiv_trajectory import evaluate, get_loaders, get_splits
from train_author_text_protocol import PairCollator, PairDataset, seed_everything
from train_stsb_frozen_mean_start import (
    assert_initial_pool_identity,
    build_frozen_b2,
)


GATE_KEYS = ("freq_gate.3.weight", "freq_gate.3.bias")
PROJ_KEYS = ("time_out_proj.weight", "time_out_proj.bias")
HEAD_KEYS = ("log_scale", "bias")
VARIANTS = {
    "step0_mean_equivalent": (),
    "head_only": ("head",),
    "gate_only": ("gate",),
    "projection_only": ("projection",),
    "pool_only": ("gate", "projection"),
    "full_step1": ("gate", "projection", "head"),
}


def clone_pool(model):
    return {key: tensor.detach().cpu().clone()
            for key, tensor in model.pool.state_dict().items()}


def clone_head(model):
    return {key: getattr(model, key).detach().cpu().clone()
            for key in HEAD_KEYS}


def copy_variant(model, pool_before, head_before, pool_after, head_after,
                 included):
    """Selectively insert the *joint-step* parameters into step-0 state."""
    selected = {key: tensor.clone() for key, tensor in pool_before.items()}
    if "gate" in included:
        for key in GATE_KEYS:
            selected[key] = pool_after[key]
    if "projection" in included:
        for key in PROJ_KEYS:
            selected[key] = pool_after[key]
    model.pool.load_state_dict(selected, strict=True)
    with torch.no_grad():
        for key in HEAD_KEYS:
            value = head_after[key] if "head" in included else head_before[key]
            getattr(model, key).copy_(value.to(
                device=getattr(model, key).device))
    model.eval()


def compare_with_e0(arrays, archive_path, label, threshold=2e-4):
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Mandatory original E0 checkpoint-prediction probe missing: "
            f"{archive_path}. E2 cannot establish exact step reproduction.")
    with np.load(archive_path, allow_pickle=False) as original:
        for key in ("labels", "score_full", "cos_full"):
            if key not in original.files or key not in arrays:
                raise RuntimeError(f"Missing {label} pair-array key {key}")
        original_labels = original["labels"]
        if not np.array_equal(original_labels, arrays["labels"]):
            raise RuntimeError(
                f"{label} E0 pair-label order mismatch; cannot compare.")
        score_difference = float(np.max(np.abs(
            original["score_full"].astype(np.float64) -
            arrays["score_full"].astype(np.float64))))
        cos_difference = float(np.max(np.abs(
            original["cos_full"].astype(np.float64) -
            arrays["cos_full"].astype(np.float64))))
    print(f"  {label}: max abs score diff={score_difference:.3e}, "
          f"cos diff={cos_difference:.3e}", flush=True)
    if score_difference > threshold or cos_difference > threshold:
        raise RuntimeError(
            f"{label} does NOT reproduce original E0. "
            f"Score diff={score_difference:.3e}, "
            f"cos diff={cos_difference:.3e} > {threshold:.1e}. "
            "STOP; do not interpret counterfactual comparisons.")
    return {"score_max_abs_difference": score_difference,
            "cos_max_abs_difference": cos_difference}


def read_e0_step1_trajectory(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Original E0 trajectory missing: {path}")
    records = [json.loads(s) for s in path.read_text(
        encoding="utf-8").splitlines() if s.strip()]
    step1 = [r for r in records if int(r["global_step"]) == 1]
    if len(step1) != 1:
        raise RuntimeError(
            f"Expected exactly one original E0 step-1 probe in {path}.")
    return step1[0]


def gate_sample_statistics(model, token_pair, device):
    """Empirical multiplier 2*sigmoid(z) over both sides of FIRST train batch.

    The normal gate is SHARED across frequency bins, not frequency-specific.
    We collect at the final Gate Linear via a forward hook; no modification.
    """
    captured = []

    def hook(_module, _inputs, logits):
        captured.append((2.0 * torch.sigmoid(logits.detach()) -
                         1.0).abs().float().cpu().reshape(-1))

    handle = model.pool.freq_gate[-1].register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            token_a, token_b, _ = token_pair
            a = {key: val.to(device) for key, val in token_a.items()}
            b = {key: val.to(device) for key, val in token_b.items()}
            model(a, b)
    finally:
        handle.remove()
    if len(captured) != 2:
        raise RuntimeError(
            "Expected two Gate invocations (one per sentence side).")
    errors = torch.cat(captured)
    return {
        "mean_absolute_gate_multiplier_minus_one": float(errors.mean()),
        "max_absolute_gate_multiplier_minus_one": float(errors.max()),
        "num_gate_coordinates_sampled": int(errors.numel()),
        "scope": "First E0 training minibatch, both sentence sides",
    }


def parameter_displacement(before, after, keys):
    sq, maxabs, num = 0.0, 0.0, 0
    for key in keys:
        difference = (after[key].double() - before[key].double())
        sq += float((difference * difference).sum())
        maxabs = max(maxabs, float(difference.abs().max()))
        num += difference.numel()
    return {"delta_l2": math.sqrt(sq),
            "delta_rms_per_parameter": math.sqrt(sq / num),
            "max_abs_parameter_delta": maxabs,
            "num_parameters": num}


def metric_delta(now, original):
    return {key: float(now[key] - original[key])
            for key in ("mse", "spearman", "pearson")}


def no_clobber_write(path, report):
    """Fixed output file; exclusive creation. No random/timestamp directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_path",
                    default="/home/data/home/wwr_lumos/models/roberta-base")
    ap.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    ap.add_argument("--pilot_dir",
                    default="outputs/text/stsb_frozen_trained_mean")
    ap.add_argument("--output_root",
                    default="outputs/text/stsb_mean_equiv_e2_step1")
    ap.add_argument("--check-only", action="store_true",
                    help="Check checkpoints/initial predictions; save NOTHING.")
    args = ap.parse_args()
    if args.seed < 0:
        ap.error("seed must be nonnegative.")

    pilot = Path(args.pilot_dir) / f"seed_{args.seed}"
    output = Path(args.output_root) / f"seed_{args.seed}" / "e2_step1.json"
    if not args.check_only and output.exists():
        raise FileExistsError(
            f"Refusing to overwrite E2 result: {output}. "
            "Send the current result for analysis before re-running.")
    summary_path = pilot / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Existing finished E0 summary required: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("seed", -1)) != args.seed:
        raise RuntimeError("E0 summary seed differs from requested seed.")
    if summary.get("frozen_roberta") is not True:
        raise RuntimeError("E0 must have a frozen trained RoBERTa encoder.")
    if (summary.get("optimizer") != "Adam" or
            abs(float(summary.get("lr", -1)) - 1e-3) > 1e-12 or
            abs(float(summary.get("weight_decay", -1))) > 1e-12):
        raise RuntimeError("E0 optimizer settings do not match E2 protocol.")
    if (summary.get("flag_pool") != "masked_mean" or
            summary.get("post_pool_norm") is not False):
        raise RuntimeError("E0 pooling protocol differs from E2.")

    reference = Path(summary["mean_reference_checkpoint"])
    if not reference.is_file():
        raise FileNotFoundError(f"Original trained Mean missing: {reference}")
    e0_step1 = read_e0_step1_trajectory(pilot / "trajectory.jsonl")
    if abs(float(e0_step1["global_step"]) - 1) > 0:
        raise RuntimeError("Incorrect E0 trajectory step.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("E2: ONE identical Adam step + parameter-spliced counterfactuals",
          flush=True)
    print(f"seed={args.seed}, device={device}, original E0={pilot}",
          flush=True)
    print("Result path (fixed, no timestamp):", output, flush=True)
    print("No old files overwritten. No additional training epochs.",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True)
    train_split, validation_split = get_splits(
        "stsb", args.stsb_data_path, "", args.seed, 0, 0)
    if (len(train_split) != int(summary["n_train"]) or
            len(validation_split) != int(summary["n_validation"])):
        raise RuntimeError("E0 dataset sizes changed.")
    train_loader, val_loader = get_loaders(
        "stsb", tokenizer, train_split, validation_split, args.seed)
    train_eval_loader = DataLoader(
        PairDataset(train_split, "stsb"), batch_size=32,
        shuffle=False, num_workers=0,
        collate_fn=PairCollator(
            tokenizer, max_length=128, label_dtype=torch.float32))

    state = torch.load(reference, map_location="cpu", weights_only=True)
    if any(k.startswith("pool.") for k in state):
        raise RuntimeError("Reference is not a trained Mean checkpoint.")
    model = build_frozen_b2(args, device, state)
    del state
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("Encoder unexpectedly trainable.")
    if not all(p.requires_grad for p in model.pool.parameters()):
        raise RuntimeError("B2 pool unexpectedly partially frozen.")

    print("[0] Strict initial Mean-equivalence checks...", flush=True)
    initial_pool_error = assert_initial_pool_identity(
        model, val_loader, device)
    model.eval()
    initial_metrics, initial_pairs = evaluate(
        model, val_loader, "stsb", device)
    initial_e0_match = compare_with_e0(
        initial_pairs, pilot / "validation_pairs_step_0.npz", "E0 Step0")
    ref_spearman = float(summary["mean_reference_validation"]["spearman"])
    if abs(initial_metrics["full"]["spearman"] - ref_spearman) > 5e-4:
        raise RuntimeError("Mean reference Spearman disagrees with E0.")
    # DataLoader generator is not consumed by validation or identity checks.
    # Iterating it now yields exactly the original E0 first shuffled batch.
    first_batch = next(iter(train_loader))
    batch_digest = hashlib.sha256(
        first_batch[0]["input_ids"].cpu().numpy().tobytes() +
        first_batch[1]["input_ids"].cpu().numpy().tobytes()
    ).hexdigest()
    print("  exact original train-loader first-batch SHA256:",
          batch_digest, flush=True)

    if args.check_only:
        print("CHECK-ONLY PASSED. No optimizer, output files or directories "
              "were created.", flush=True)
        return

    pool_before, head_before = clone_pool(model), clone_head(model)
    print("[1] Reproducing E0's Adam Step 1 (one batch, one update)...",
          flush=True)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=0.0)
    seed_everything(args.seed + 100000)
    model.train()
    tok_a, tok_b, y = first_batch
    a = {k: v.to(device) for k, v in tok_a.items()}
    b = {k: v.to(device) for k, v in tok_b.items()}
    target = y.to(device) / 5.0
    optimizer.zero_grad(set_to_none=True)
    predicted, _ = model(a, b)
    batch_loss = F.mse_loss(predicted, target)
    batch_loss.backward()

    # Verify initial zero-output Gate correctly blocks all upstream
    # derivatives; only Gate output, Projection and Head can move at step 1.
    permitted = {
        "pool." + k for k in (*GATE_KEYS, *PROJ_KEYS)
    } | set(HEAD_KEYS)
    zero_upstream = []
    for name, p in model.named_parameters():
        if not p.requires_grad or name in permitted:
            continue
        maximum = (0.0 if p.grad is None
                   else float(p.grad.detach().abs().max()))
        if maximum > 1e-9:
            raise RuntimeError(
                f"Unexpected nonzero first-step upstream gradient: "
                f"{name}={maximum:.3e}")
        zero_upstream.append(name)
    optimizer.step()  # THE ONLY parameter update in E2.
    model.eval()
    pool_after, head_after = clone_pool(model), clone_head(model)
    for key in pool_before:
        if key not in (*GATE_KEYS, *PROJ_KEYS):
            if not torch.equal(pool_before[key], pool_after[key]):
                raise RuntimeError(
                    f"Unexpected extra parameter change at Step 1: {key}")
    print(f"  first batch MSE before update: {float(batch_loss):.8f}",
          flush=True)

    print("[2] Checking exact Step-1 reproduction vs saved E0...", flush=True)
    full_metrics, full_pairs = evaluate(model, val_loader, "stsb", device)
    step1_e0_match = compare_with_e0(
        full_pairs, pilot / "validation_pairs_step_1.npz", "E0 Step1")
    expected_step1 = e0_step1["full_flag_b2"]
    if (abs(full_metrics["full"]["spearman"] -
            float(expected_step1["spearman"])) > 5e-4 or
            abs(full_metrics["full"]["mse"] -
                float(expected_step1["mse"])) > 5e-5):
        raise RuntimeError(
            "Step-1 metrics differ from original E0. STOP.")

    print("[3] Evaluating six joint-update parameter-spliced variants...",
          flush=True)
    rows = {}
    init_train, init_val = None, None
    init_cos = None
    for name, included in VARIANTS.items():
        copy_variant(model, pool_before, head_before, pool_after, head_after,
                     included)
        val_metric, val_arr = evaluate(model, val_loader, "stsb", device)
        train_metric, train_arr = evaluate(
            model, train_eval_loader, "stsb", device)
        gate_info = gate_sample_statistics(model, first_batch, device)

        if name == "step0_mean_equivalent":
            init_train = train_metric["full"]
            init_val = val_metric["full"]
            init_cos = val_arr["cos_full"].astype(np.float64).copy()
        row = {
            "module_values_from_joint_step1": list(included),
            "train": train_metric["full"],
            "validation": val_metric["full"],
            "train_mean_abs_cos_delta_vs_same_encoder_mean":
                train_metric["mean_abs_pair_cos_delta"],
            "validation_mean_abs_cos_delta_vs_same_encoder_mean":
                val_metric["mean_abs_pair_cos_delta"],
            "val_mean_abs_pair_cos_change_vs_step0": float(np.mean(
                np.abs(val_arr["cos_full"].astype(np.float64) -
                       init_cos))),
            "val_max_abs_pair_cos_change_vs_step0": float(np.max(
                np.abs(val_arr["cos_full"].astype(np.float64) -
                       init_cos))),
            "gate_multiplier_deviation": gate_info,
            "train_delta_vs_step0": metric_delta(
                train_metric["full"], init_train),
            "validation_delta_vs_step0": metric_delta(
                val_metric["full"], init_val),
        }
        rows[name] = row
        print(f"  {name:<22} "
              f"train_mse={row['train']['mse']:.7f} "
              f"val_mse={row['validation']['mse']:.7f} "
              f"val_sp={row['validation']['spearman']:.6f} "
              f"Δval_sp={row['validation_delta_vs_step0']['spearman']:+.6f}",
              flush=True)

    # Cross-check that temporarily splicing parameters has not changed
    # saved joint-step results.
    full_vs_saved_sp = abs(
        rows["full_step1"]["validation"]["spearman"] -
        full_metrics["full"]["spearman"])
    if full_vs_saved_sp > 1e-7:
        raise RuntimeError("Full variant reconstruction did not match E0.")
    baseline_vs_initial_sp = abs(
        rows["step0_mean_equivalent"]["validation"]["spearman"] -
        initial_metrics["full"]["spearman"])
    if baseline_vs_initial_sp > 1e-7:
        raise RuntimeError("Restored initial variant does not match Step0.")

    report = {
        "experiment": "E2_joint_Adam_step1_parameter_splice",
        "seed": args.seed,
        "hypothesis": "Does the actual first Adam step harm validation; "
                      "which module's joint-step parameter changes "
                      "are necessary/sufficient for local damage?",
        "original_e0_summary": str(summary_path),
        "original_e0_step1_array":
            str(pilot / "validation_pairs_step_1.npz"),
        "reference_mean_checkpoint": str(reference),
        "device": str(device),
        "torch_version": torch.__version__,
        "train_pairs": len(train_split),
        "validation_pairs": len(validation_split),
        "train_eval_batch_size": 32,
        "validation_eval_batch_size": 64,
        "model_initial_identity_pool_max_abs_diff": initial_pool_error,
        "e0_reproduction": {
            "step0": initial_e0_match,
            "step1": step1_e0_match,
            "original_e0_step1_validation": expected_step1,
        },
        "first_train_minibatch": {
            "pairs": len(y),
            "token_id_sha256": batch_digest,
            "mse_before_step1": float(batch_loss.detach().cpu()),
        },
        "optimizer": "torch.optim.Adam(lr=1e-3, weight_decay=0.0)",
        "optimizer_steps_in_memory": 1,
        "frozen_roberta": True,
        "zero_initial_upstream_gradients_verified":
            zero_upstream,
        "gate_output_displacement": parameter_displacement(
            pool_before, pool_after, GATE_KEYS),
        "projection_displacement": parameter_displacement(
            pool_before, pool_after, PROJ_KEYS),
        "head_displacement": parameter_displacement(
            head_before, head_after, HEAD_KEYS),
        "variants": rows,
        "limitations": [
            "Only first actual Adam step, no multi-epoch causal claim.",
            "Each variant splices parameter blocks from ONE JOINT step, "
            "not separate module-only training runs.",
            "Module effects need not add due to nonlinear interactions.",
            "Train evaluation uses deterministic no-shuffle dynamic "
            "padding; E0 online train MSE used shuffled batches.",
            "The E0 validation split selected the reference checkpoint; "
            "it is not an untouched test set.",
            "Head affine updates can change MSE but not cosine ranking "
            "or Spearman (except tiny numerical effects).",
            "Gate multiplier deviation sampled on the same first train "
            "batch; it is not a dataset-wide Gate statistic.",
        ],
    }
    no_clobber_write(output, report)
    print(f"\nSaved E2 JSON (fixed path): {output}", flush=True)


if __name__ == "__main__":
    main()
