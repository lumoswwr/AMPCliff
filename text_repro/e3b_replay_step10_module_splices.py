#!/usr/bin/env python3
"""E3b: reproducible E0 Step-10 Adam replay and factorial parameter splices.

This is not eight separately trained models. Train ONE jointly optimized B2
for EXACTLY 10 E0 steps; test all 2x2x2 combinations of initial/Step-10
Gate+latent, Projection, Head blocks. Replay MUST reproduce E0's saved
per-pair predictions at Steps 0, 1, and 10 or the experiment stops.

No E0 checkpoint or prior output may be overwritten. A --check-only run
checks E0 inputs and initialization but writes no E3b result.
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
from scipy.stats import rankdata
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from e2_adam_step1_counterfactual import (
    clone_head,
    clone_pool,
    compare_with_e0,
    gate_sample_statistics,
    metric_delta,
    parameter_displacement,
)
from explore_mean_equiv_trajectory import evaluate, get_loaders, get_splits
from train_author_text_protocol import PairCollator, PairDataset, seed_everything
from train_stsb_frozen_mean_start import (
    assert_initial_pool_identity,
    build_frozen_b2,
)


HEAD_KEYS = ("log_scale", "bias")
PROJECTION_KEYS = ("time_out_proj.weight", "time_out_proj.bias")
VARIANTS = {
    "step0_mean_equivalent": (),
    "head_only": ("head",),
    "gate_latent_only": ("gate_latent",),
    "projection_only": ("projection",),
    "pool_only": ("gate_latent", "projection"),
    "gate_latent_plus_head": ("gate_latent", "head"),
    "projection_plus_head": ("projection", "head"),
    "full_step10": ("gate_latent", "projection", "head"),
}


def load_e0_metadata(pilot, seed):
    summary_path = pilot / "summary.json"
    trajectory_path = pilot / "trajectory.jsonl"
    if not summary_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError(
            "E3b requires existing E0 summary.json and trajectory.jsonl.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (int(summary.get("seed", -1)) != seed
            or summary.get("frozen_roberta") is not True):
        raise RuntimeError("E0 seed/frozen RoBERTa mismatch.")
    if (summary.get("optimizer") != "Adam"
            or abs(float(summary.get("lr", -1.0)) - 1e-3) > 1e-12
            or abs(float(summary.get("weight_decay", -1.0))) > 1e-12):
        raise RuntimeError("E0 Adam settings are different.")
    if (summary.get("flag_pool") != "masked_mean"
            or summary.get("post_pool_norm") is not False
            or summary.get("projection_init") != "identity"
            or abs(float(summary.get("gate_init", -1.0)) - 1.0) > 1e-12):
        raise RuntimeError("E0 B2 pooling protocol is different.")
    records = [json.loads(line) for line in trajectory_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    probes = {}
    for step in (0, 1, 10):
        matched = [row for row in records if int(row["global_step"]) == step]
        if len(matched) != 1:
            raise RuntimeError(
                f"Exactly one E0 trajectory probe required for Step {step}.")
        path = pilot / f"validation_pairs_step_{step}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"E0 Step-{step} predictions missing: {path}")
        probes[step] = matched[0]
    reference = Path(summary["mean_reference_checkpoint"])
    if not reference.is_file():
        raise FileNotFoundError(
            f"Original E0 trained Mean checkpoint missing: {reference}")
    return summary_path, summary, probes, reference


def identify_pool_groups(model):
    pool_names = set(model.pool.state_dict())
    projection = set(PROJECTION_KEYS)
    if not projection.issubset(pool_names):
        raise RuntimeError("E0 Projection keys absent from current B2.")
    # Every other pool parameter or persistent buffer belongs to the
    # Gate+Latent pathway, including attention, FFN, and gate input/output.
    # Some modules such as unused norm3 will remain unchanged.
    gate_latent = pool_names - projection
    if not {"freq_gate.3.weight", "freq_gate.3.bias",
            "freq_gate.0.weight", "attn.in_proj_weight",
            "latents"}.issubset(gate_latent):
        raise RuntimeError("Gate/Latent parameter names changed.")
    model_trainable = {
        name for name, p in model.named_parameters() if p.requires_grad}
    expected_trainable = {
        "pool." + key for key in pool_names
    } | set(HEAD_KEYS)
    if model_trainable != expected_trainable:
        raise RuntimeError(
            "B2 trainable parameter set changed: unexpected "
            f"{sorted(model_trainable ^ expected_trainable)[:20]}")
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("Encoder was not frozen.")
    return sorted(gate_latent), sorted(projection)


def apply_variant(model, initial_pool, final_pool, initial_head,
                  final_head, included, gate_keys, projection_keys):
    chosen = {}
    for key in initial_pool:
        group = "projection" if key in projection_keys else "gate_latent"
        chosen[key] = (final_pool[key] if group in included
                       else initial_pool[key])
    model.pool.load_state_dict(chosen, strict=True)
    with torch.no_grad():
        for key in HEAD_KEYS:
            head = final_head if "head" in included else initial_head
            getattr(model, key).copy_(head[key].to(
                getattr(model, key).device))
    model.eval()


def pair_diff(arrays, base, original_labels):
    current_cos = arrays["cos_full"].astype(np.float64)
    baseline_cos = base["cos_full"].astype(np.float64)
    if not np.array_equal(original_labels, arrays["labels"]):
        raise RuntimeError("Validation ordering changed between variants.")
    shift = np.abs(current_cos - baseline_cos)
    rank_shift = np.abs(rankdata(current_cos) - rankdata(baseline_cos))
    return {
        "mean_absolute_pair_cos_change_vs_step0": float(shift.mean()),
        "max_absolute_pair_cos_change_vs_step0": float(shift.max()),
        "mean_absolute_validation_rank_shift_vs_step0":
            float(rank_shift.mean()),
        "fraction_validation_pairs_shifted_at_least_50_ranks":
            float((rank_shift >= 50).mean()),
    }


def numeric_check(metrics, expected, step):
    for key, eps in (("spearman", 5e-5), ("mse", 5e-6)):
        difference = abs(float(metrics[key]) -
                         float(expected["full_flag_b2"][key]))
        if difference > eps:
            raise RuntimeError(
                f"E0 Step {step} {key} changed by {difference:.3e}. STOP.")


def changed_parameter_counts(pool0, pool10, keys):
    altered = [key for key in keys if not torch.equal(
        pool0[key], pool10[key])]
    return {"changed_tensors": len(altered), "total_tensors": len(keys),
            "names": altered}


def interaction(summary_rows, metric, split, with_head):
    """Difference-in-differences; descriptive, especially for Spearman."""
    if with_head:
        a, b, c, d = ("head_only", "gate_latent_plus_head",
                      "projection_plus_head", "full_step10")
    else:
        a, b, c, d = ("step0_mean_equivalent", "gate_latent_only",
                      "projection_only", "pool_only")
    return float(summary_rows[d][split][metric] -
                 summary_rows[b][split][metric] -
                 summary_rows[c][split][metric] +
                 summary_rows[a][split][metric])


def write_exclusive(path, payload):
    if path.exists():
        raise FileExistsError(f"E3b result exists, refusing overwrite: {path}")
    # Serialize first so invalid/NaN values do not create a partial file.
    content = json.dumps(payload, indent=2, ensure_ascii=False,
                         allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_path",
                        default="/home/data/home/wwr_lumos/models/roberta-base")
    parser.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    parser.add_argument("--pilot_dir",
                        default="outputs/text/stsb_frozen_trained_mean")
    parser.add_argument("--output_root",
                        default="outputs/text/stsb_mean_equiv_e3b_step10")
    parser.add_argument("--check-only", action="store_true",
                        help="Check inputs and Step0 only, produce NO results.")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative.")
    pilot = Path(args.pilot_dir) / f"seed_{args.seed}"
    output = (Path(args.output_root) / f"seed_{args.seed}" /
              "e3b_step10_counterfactual.json")
    if output.exists() and not args.check_only:
        raise FileExistsError(f"E3b already exists: {output}")

    summary_path, summary, original, reference = load_e0_metadata(
        pilot, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("E3b: replay E0's first 10 Adam steps and splice all module blocks",
          flush=True)
    print(f"seed={args.seed} device={device}; fixed result={output}",
          flush=True)
    print("NO old checkpoints/outputs overwritten; ONLY 10 optimizer steps.",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True)
    train_data, val_data = get_splits(
        "stsb", args.stsb_data_path, "", args.seed, 0, 0)
    if (len(train_data) != int(summary["n_train"]) or
            len(val_data) != int(summary["n_validation"])):
        raise RuntimeError("E0 train/validation sizes mismatch.")
    train_loader, val_loader = get_loaders(
        "stsb", tokenizer, train_data, val_data, args.seed)
    # All variants evaluated on the same deterministic train order.
    train_eval_loader = DataLoader(
        PairDataset(train_data, "stsb"), batch_size=32, shuffle=False,
        num_workers=0,
        collate_fn=PairCollator(
            tokenizer, max_length=128, label_dtype=torch.float32))

    state = torch.load(reference, map_location="cpu", weights_only=True)
    if any(key.startswith("pool.") for key in state):
        raise RuntimeError("Reference checkpoint is not a Mean state_dict.")
    model = build_frozen_b2(args, device, state)
    del state
    gate_keys, projection_keys = identify_pool_groups(model)

    print("[0] Check initial Mean identity and original E0 Step 0...", flush=True)
    pool_identity = assert_initial_pool_identity(model, val_loader, device)
    model.eval()
    initial_metrics, initial_arrays = evaluate(
        model, val_loader, "stsb", device)
    replay_matches = {
        "step0": compare_with_e0(
            initial_arrays, pilot / "validation_pairs_step_0.npz", "E0 Step0")
    }
    numeric_check(initial_metrics["full"], original[0], 0)
    if args.check_only:
        print("CHECK-ONLY PASSED: no Step 1-10 training, no E3b results.",
              flush=True)
        return

    pool0, head0 = clone_pool(model), clone_head(model)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=0.0)
    seed_everything(args.seed + 100000)
    model.train()
    first_batch_digest = None
    first_batch = None
    online_losses = []

    print("[1] Replay E0's first 10 training minibatches...", flush=True)
    for step, (tok_a, tok_b, label) in enumerate(train_loader, start=1):
        if step > 10:
            break
        if step == 1:
            first_batch = (tok_a, tok_b, label)
            first_batch_digest = hashlib.sha256(
                tok_a["input_ids"].numpy().tobytes() +
                tok_b["input_ids"].numpy().tobytes()).hexdigest()
        a = {key: value.to(device) for key, value in tok_a.items()}
        b = {key: value.to(device) for key, value in tok_b.items()}
        target = label.to(device) / 5.0
        opt.zero_grad(set_to_none=True)
        predicted, _ = model(a, b)
        loss = F.mse_loss(predicted, target)
        loss.backward()
        opt.step()
        online_losses.append(float(loss.detach().cpu()))
        if step in (1, 10):
            model.eval()
            val_metrics, arrays = evaluate(model, val_loader, "stsb", device)
            replay_matches[f"step{step}"] = compare_with_e0(
                arrays, pilot / f"validation_pairs_step_{step}.npz",
                f"E0 Step{step}")
            numeric_check(val_metrics["full"], original[step], step)
            print(f"  Verified E0 Step {step}: "
                  f"val_sp={val_metrics['full']['spearman']:.6f} "
                  f"val_mse={val_metrics['full']['mse']:.7f}", flush=True)
            model.train()

    if len(online_losses) != 10:
        raise RuntimeError("Less than 10 E0 training batches available.")
    if abs(online_losses[0] -
           float(original[1]["last_minibatch_train_mse"])) > 1e-7:
        raise RuntimeError("First batch loss differs from E0.")
    if abs(online_losses[9] -
           float(original[10]["last_minibatch_train_mse"])) > 1e-7:
        raise RuntimeError("Tenth batch loss differs from E0.")
    print("  All Step-1 and Step-10 per-pair replay checks passed.",
          flush=True)

    pool10, head10 = clone_pool(model), clone_head(model)
    gate_changes = changed_parameter_counts(pool0, pool10, gate_keys)
    proj_changes = changed_parameter_counts(pool0, pool10, projection_keys)
    if not gate_changes["changed_tensors"] or not proj_changes["changed_tensors"]:
        raise RuntimeError("Expected both Gate+Latent and Projection to update.")

    print("[2] Evaluating 8 factorial Step-10 parameter splices...",
          flush=True)
    baseline_train, baseline_val = None, None
    rows = {}
    for name, included in VARIANTS.items():
        apply_variant(model, pool0, pool10, head0, head10,
                      included, gate_keys, projection_keys)
        val_metrics, val_arrays = evaluate(
            model, val_loader, "stsb", device)
        train_metrics, _train_arrays = evaluate(
            model, train_eval_loader, "stsb", device)
        if baseline_val is None:
            baseline_train = train_metrics["full"]
            baseline_val = val_metrics["full"]
        row = {
            "parameter_blocks_at_step10": list(included),
            "train": train_metrics["full"],
            "validation": val_metrics["full"],
            "train_delta_vs_step0":
                metric_delta(train_metrics["full"], baseline_train),
            "validation_delta_vs_step0":
                metric_delta(val_metrics["full"], baseline_val),
            "validation_cosine_vs_step0":
                pair_diff(val_arrays, initial_arrays,
                          initial_arrays["labels"]),
            "mean_abs_val_cos_difference_from_mean_at_same_head":
                val_metrics["mean_abs_pair_cos_delta"],
        }
        rows[name] = row
        print(f"  {name:<25} "
              f"val_sp={row['validation']['spearman']:.6f} "
              f"Δsp={row['validation_delta_vs_step0']['spearman']:+.6f} "
              f"val_mse={row['validation']['mse']:.7f} "
              f"train_mse={row['train']['mse']:.7f}",
              flush=True)

    # End-to-end equivalence checks AFTER splicing, not just during replay.
    for variant, step in (
        ("step0_mean_equivalent", 0), ("full_step10", 10)
    ):
        target = original[step]["full_flag_b2"]
        actual = rows[variant]["validation"]
        for key in ("spearman", "mse", "pearson"):
            if abs(actual[key] - float(target[key])) > 5e-6:
                raise RuntimeError(
                    f"Splice {variant} diverged from E0 Step {step} {key}.")
    # Positive affine Head alone must not affect rank-based Spearman.
    for a, b in [
        ("step0_mean_equivalent", "head_only"),
        ("gate_latent_only", "gate_latent_plus_head"),
        ("projection_only", "projection_plus_head"),
        ("pool_only", "full_step10"),
    ]:
        if abs(rows[a]["validation"]["spearman"] -
               rows[b]["validation"]["spearman"]) > 1e-6:
            raise RuntimeError(
                f"Head unexpectedly affected Spearman: {a} vs {b}.")

    print("[3] Compute block effects at common Step-10 Head...", flush=True)
    shared_head_effects = {
        "gate_path_change_with_initial_projection": {
            key: (rows["gate_latent_plus_head"]["validation"][key] -
                  rows["head_only"]["validation"][key])
            for key in ("spearman", "mse")
        },
        "projection_change_with_initial_gate_path": {
            key: (rows["projection_plus_head"]["validation"][key] -
                  rows["head_only"]["validation"][key])
            for key in ("spearman", "mse")
        },
        "increment_of_gate_path_given_step10_projection": {
            key: (rows["full_step10"]["validation"][key] -
                  rows["projection_plus_head"]["validation"][key])
            for key in ("spearman", "mse")
        },
        "increment_of_projection_given_step10_gate_path": {
            key: (rows["full_step10"]["validation"][key] -
                  rows["gate_latent_plus_head"]["validation"][key])
            for key in ("spearman", "mse")
        },
        "interaction_difference_in_differences": {
            key: interaction(rows, key, "validation", True)
            for key in ("spearman", "mse")
        },
    }
    # Restoring exactly Step10 leaves gate multiplier hook safe and
    # unaffected; gate deviation evaluated on the first training batch.
    apply_variant(model, pool0, pool10, head0, head10,
                  VARIANTS["full_step10"], gate_keys, projection_keys)
    gate_snapshot = gate_sample_statistics(model, first_batch, device)

    report = {
        "experiment": "E3b_Adam_10step_replay_and_factorial_splices",
        "seed": args.seed,
        "purpose": (
            "Identify which JOINTLY learned Gate+Latent and Projection "
            "parameter blocks account for Step-10 validation ranking "
            "change relative to Mean, with Head held fixed."),
        "original_e0_summary": str(summary_path),
        "reference_mean_checkpoint": str(reference),
        "original_e0_step10_array":
            str(pilot / "validation_pairs_step_10.npz"),
        "device": str(device),
        "torch_version": torch.__version__,
        "optimizer_steps_replayed": len(online_losses),
        "optimizer": "Adam, learning_rate=1e-3, weight_decay=0",
        "frozen_encoder": True,
        "dataset": "STSB train / validation only",
        "train_pairs": len(train_data),
        "validation_pairs": len(val_data),
        "first_batch_token_sha256": first_batch_digest,
        "online_minibatch_losses": online_losses,
        "initial_pool_max_abs_difference_from_mean": pool_identity,
        "replay_vs_e0_pair_predictions": replay_matches,
        "block_definitions": {
            "gate_latent": "ALL pool state_dict keys EXCEPT "
                           "time_out_proj.weight/bias; includes "
                           "gate input/output, latents, Attention, FFN, "
                           "normalization, and unchanged/unused parameters.",
            "projection": list(projection_keys),
            "head": list(HEAD_KEYS),
            "splices": "Use the block values from one 10-step JOINT "
                       "Adam training run; no separate frozen-block runs.",
        },
        "changed_gate_latent_tensors": gate_changes,
        "changed_projection_tensors": proj_changes,
        "gate_latent_parameter_displacement": parameter_displacement(
            pool0, pool10, gate_keys),
        "projection_parameter_displacement": parameter_displacement(
            pool0, pool10, projection_keys),
        "head_parameter_displacement": parameter_displacement(
            head0, head10, HEAD_KEYS),
        "full_step10_gate_deviation_first_train_batch": gate_snapshot,
        "variants": rows,
        "shared_step10_head_effects": shared_head_effects,
        "limitations": [
            "The 8 variants are parameter splices from ONE joint "
            "trajectory; they do not identify what would happen if "
            "Gate-only or Projection-only were trained separately.",
            "All Step10 comparisons concern frozen encoder, same "
            "existing validation set, and one seed only.",
            "Spearman is nonlinear/discontinuous in order swaps; "
            "interaction difference-in-differences is not proof "
            "of a specific architectural synergy.",
            "A changed Gate+Latent block includes many interdependent "
            "upstream layers. It is not isolated FFT-vs-attention mechanism.",
            "Full-training MSE from deterministic no-shuffle "
            "evaluation is distinct from online minibatch MSE.",
            "Mean reference was selected using this validation split; "
            "the result is not an independent held-out test.",
        ],
    }
    write_exclusive(output, report)
    print(f"\nSaved FIXED E3b result: {output}", flush=True)


if __name__ == "__main__":
    main()
