#!/usr/bin/env python3
"""E4: five-seed functional representation drift from exact trained Mean.

E3b measured per-pair cosine changes after splicing joint-trajectory
parameter blocks. E4 measures NEW quantities: per-sentence normalized
embedding angular drift (Gate+Latent vs Projection), conditional effects,
and per-pair squared-error improvement with the SAME current score Head.

The ONLY training is exact replay of E0's first 10 joint Adam minibatches
per seed. It verifies E0 Step 0/1/10 predictions. ALL seeds 0..4 are run
in one command. One fixed JSON result; never overwrite existing outputs.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from transformers import AutoTokenizer

from e2_adam_step1_counterfactual import compare_with_e0
from e3b_replay_step10_module_splices import (
    identify_pool_groups, load_e0_metadata, numeric_check,
)
from explore_mean_equiv_trajectory import (
    evaluate, get_loaders, get_splits, metrics,
)
from train_author_text_protocol import seed_everything
from train_stsb_frozen_mean_start import (
    assert_initial_pool_identity, build_frozen_b2,
)
from AMPCliff.factory.pooling.llm_pooling_dropin import masked_mean_pooling


SEEDS = (0, 1, 2, 3, 4)
VARIANTS = ("mean", "gate_latent_only", "projection_only", "full_step10")
ANGLE_CONTRASTS = {
    "gate_latent_vs_mean": ("gate_latent_only", "mean"),
    "projection_vs_mean": ("projection_only", "mean"),
    "full_vs_mean": ("full_step10", "mean"),
    "projection_after_gate": ("full_step10", "gate_latent_only"),
    "gate_after_projection": ("full_step10", "projection_only"),
}


def preflight(args):
    sources = {}
    for seed in SEEDS:
        pilot = Path(args.pilot_dir) / f"seed_{seed}"
        summary_path, summary, probes, reference = load_e0_metadata(
            pilot, seed)
        # E0 and E3b refer to the same 5749/1500 STSB protocol.
        if int(summary["n_train"]) != 5749 or int(
                summary["n_validation"]) != 1500:
            raise RuntimeError(f"Seed {seed}: unexpected STSB dataset size.")
        with np.load(pilot / "validation_pairs_step_10.npz",
                     allow_pickle=False) as npz:
            if len(npz["labels"]) != int(summary["n_validation"]):
                raise RuntimeError(f"Seed {seed}: E0 NPZ size changed.")
        sources[seed] = (pilot, summary_path, summary, probes, reference)
    return sources


def angle_degrees(a, b):
    """Stable normalized-vector angle, even when almost identical.

    theta = 2 atan2(||a-b||, ||a+b||); no arccos near 1.
    """
    if a.shape != b.shape:
        raise RuntimeError("Embedding shape mismatch.")
    numerator = torch.linalg.vector_norm(a - b, dim=-1)
    denominator = torch.linalg.vector_norm(a + b, dim=-1)
    return torch.rad2deg(2.0 * torch.atan2(numerator, denominator))


def vector_alignment_stats(parts):
    ang = np.concatenate(parts).astype(np.float64)
    if len(ang) != 3000 or not np.isfinite(ang).all():
        raise RuntimeError("Expected 3000 finite validation sentence occurrences.")
    return {
        "sentence_occurrences": int(ang.size),
        "mean_angle_degrees": float(ang.mean()),
        "median_angle_degrees": float(np.median(ang)),
        "p90_angle_degrees": float(np.percentile(ang, 90)),
        "fraction_angle_above_5_deg": float(np.mean(ang > 5)),
        "fraction_angle_above_10_deg": float(np.mean(ang > 10)),
    }


@torch.no_grad()
def evaluate_representation_drift(model, loader, device):
    """Same frozen encoder + same Step10 Head for all variants.

    Gate-only is the Step10 Gate+Latent PRE-projection vector.
    Projection-only applies the trained Step10 affine projection to Mean.
    Full composes the two. No separate optimization or weight splicing.
    """
    model.eval()
    labels_chunks = []
    cos_chunks = {v: [] for v in VARIANTS}
    score_chunks = {v: [] for v in VARIANTS}
    angle_chunks = {k: [] for k in ANGLE_CONTRASTS}
    composition_error_max = 0.
    for tok_a, tok_b, labels in loader:
        labels_chunks.append(labels.numpy().astype(np.float32))
        sides = []
        for tokens in (tok_a, tok_b):
            tokens = {k: val.to(device) for k, val in tokens.items()}
            mask = tokens["attention_mask"]
            hidden = model.encoder(**tokens).last_hidden_state
            mean_raw = masked_mean_pooling(hidden, mask, eps=1e-6)
            full_raw = model.pool(hidden, attention_mask=mask)
            gate_raw = model.pool._last_pooled_pre_projection
            projection_only_raw = model.pool.time_out_proj(mean_raw)
            expected_full = model.pool.time_out_proj(gate_raw)
            composition_error_max = max(
                composition_error_max,
                float((full_raw - expected_full).abs().max().item()),
            )
            if composition_error_max > 2e-6:
                raise RuntimeError(
                    "B2 protocol changed (dropout, norm or Projection): "
                    "full is not Projection(gated preprojection mean)."
                )

            vec = {
                "mean": F.normalize(mean_raw, p=2, dim=-1),
                "gate_latent_only": F.normalize(gate_raw, p=2, dim=-1),
                "projection_only": F.normalize(
                    projection_only_raw, p=2, dim=-1),
                "full_step10": F.normalize(full_raw, p=2, dim=-1),
            }
            for contrast, (a, b) in ANGLE_CONTRASTS.items():
                angle_chunks[contrast].append(
                    angle_degrees(vec[a], vec[b]).cpu().numpy())
            sides.append(vec)

        for name in VARIANTS:
            cosine = (sides[0][name] * sides[1][name]).sum(dim=-1)
            score = model.log_scale.exp() * cosine + model.bias
            cos_chunks[name].append(
                cosine.detach().cpu().numpy().astype(np.float32))
            score_chunks[name].append(
                score.detach().cpu().numpy().astype(np.float32))

    labels = np.concatenate(labels_chunks)
    cosine = {k: np.concatenate(v) for k, v in cos_chunks.items()}
    scores = {k: np.concatenate(v) for k, v in score_chunks.items()}
    if len(labels) != 1500:
        raise RuntimeError("Validation has a non-E0 pair count.")
    angles = {name: vector_alignment_stats(parts)
              for name, parts in angle_chunks.items()}

    ref_cos = cosine["mean"].astype(np.float64)
    ref_rank = rankdata(ref_cos)
    same_head_ref_error = (
        scores["mean"].astype(np.float64) -
        labels.astype(np.float64) / 5.
    ) ** 2
    variants = {}
    for name in VARIANTS:
        score = scores[name].astype(np.float64)
        current_cos = cosine[name].astype(np.float64)
        loss = (score - labels.astype(np.float64) / 5.) ** 2
        loss_delta = loss - same_head_ref_error
        rank_diff = np.abs(rankdata(current_cos) - ref_rank)
        variants[name] = {
            "validation": metrics("stsb", labels, score),
            "pair_similarity_drift_vs_mean": {
                "mean_abs_cosine_delta": float(np.mean(np.abs(
                    current_cos - ref_cos))),
                "p90_abs_cosine_delta": float(np.percentile(np.abs(
                    current_cos - ref_cos), 90)),
                "mean_abs_rank_shift": float(rank_diff.mean()),
                "fraction_rank_shift_at_least_50": float(
                    np.mean(rank_diff >= 50)),
            },
            "prediction_vs_mean_with_identical_step10_head": {
                "mean_squared_error_delta": float(loss_delta.mean()),
                "fraction_pairs_squared_error_improved":
                    float(np.mean(loss_delta < -1e-10)),
                "fraction_pairs_squared_error_worsened":
                    float(np.mean(loss_delta > 1e-10)),
                "fraction_pairs_essentially_unchanged":
                    float(np.mean(np.abs(loss_delta) <= 1e-10)),
            },
        }
    return labels, cosine, scores, angles, variants, composition_error_max


def e3b_crosscheck(args, seed, variants):
    path = (Path(args.e3b_root) / f"seed_{seed}" /
            "e3b_step10_counterfactual.json")
    if not path.is_file():
        return {"status": "no_E3b_JSON_present",
                "note": "E0 Step0/1/10 predictions still checked."}
    previous = json.loads(path.read_text(encoding="utf-8"))
    mapping = {
        "mean": "head_only",
        "gate_latent_only": "gate_latent_plus_head",
        "projection_only": "projection_plus_head",
        "full_step10": "full_step10",
    }
    largest_diff = 0.
    for variant, prev in mapping.items():
        for key in ("spearman", "mse", "pearson"):
            d = abs(float(variants[variant]["validation"][key]) -
                    float(previous["variants"][prev]["validation"][key]))
            largest_diff = max(largest_diff, d)
            if d > 5e-6:
                raise RuntimeError(
                    f"E4 and archived E3b mismatch seed={seed}, "
                    f"{variant}/{key}: diff={d:.3g}; STOP.")
    return {"status": "passed",
            "max_metric_abs_difference": largest_diff,
            "source": str(path)}


def replay_seed(args, seed, tokenizer, sources, device):
    pilot, summary_path, summary, probes, reference = sources[seed]
    print(f"\n[E4 seed {seed}] exact E0 replay, 10 Adam steps", flush=True)
    args.seed = seed
    train_split, val_split = get_splits(
        "stsb", args.stsb_data_path, "", seed, 0, 0)
    if len(train_split) != int(summary["n_train"]) or len(
            val_split) != int(summary["n_validation"]):
        raise RuntimeError(f"Seed {seed}: dataset sizes changed.")
    train_loader, val_loader = get_loaders(
        "stsb", tokenizer, train_split, val_split, seed)
    reference_state = torch.load(
        reference, map_location="cpu", weights_only=True)
    if any(k.startswith("pool.") for k in reference_state):
        raise RuntimeError("Expected Mean checkpoint, found pool state.")
    model = build_frozen_b2(args, device, reference_state)
    del reference_state
    gate_keys, projection_keys = identify_pool_groups(model)
    identity_err = assert_initial_pool_identity(model, val_loader, device)

    model.eval()
    eval0, arrays0 = evaluate(model, val_loader, "stsb", device)
    checks = {
        "step0": compare_with_e0(
            arrays0, pilot / "validation_pairs_step_0.npz", "E0 Step0")
    }
    numeric_check(eval0["full"], probes[0], 0)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=0.)
    seed_everything(seed + 100000)
    model.train()
    loss_list = []
    for step, (tok_a, tok_b, y) in enumerate(train_loader, 1):
        if step > 10:
            break
        a = {k: v.to(device) for k, v in tok_a.items()}
        b = {k: v.to(device) for k, v in tok_b.items()}
        optimizer.zero_grad(set_to_none=True)
        pred, _ = model(a, b)
        loss = F.mse_loss(pred, y.to(device) / 5.)
        loss.backward()
        optimizer.step()
        loss_list.append(float(loss.detach().cpu()))
        if step in (1, 10):
            model.eval()
            ev, arr = evaluate(model, val_loader, "stsb", device)
            checks[f"step{step}"] = compare_with_e0(
                arr, pilot / f"validation_pairs_step_{step}.npz",
                f"E0 Step{step} seed{seed}")
            numeric_check(ev["full"], probes[step], step)
            model.train()
    if len(loss_list) != 10:
        raise RuntimeError("Expected exactly 10 online Adam batches.")
    for step, idx in ((1, 0), (10, 9)):
        if abs(loss_list[idx] -
               float(probes[step]["last_minibatch_train_mse"])) > 1e-7:
            raise RuntimeError(f"E0 online batch mismatch seed{seed}/Step{step}")

    labels, cosine, scores, angles, variants, algebra_err = (
        evaluate_representation_drift(model, val_loader, device))
    arrays10 = {"labels": labels, "score_full": scores["full_step10"],
                "cos_full": cosine["full_step10"]}
    checks["step10_new_vector_path"] = compare_with_e0(
        arrays10, pilot / "validation_pairs_step_10.npz",
        f"E0 Step10 vector path seed{seed}")
    for key in ("spearman", "mse"):
        if abs(variants["full_step10"]["validation"][key] -
               float(probes[10]["full_flag_b2"][key])) > 5e-6:
            raise RuntimeError(
                f"E0 per-pair vector path metric mismatch seed{seed}/{key}")
    checks["e3b"] = e3b_crosscheck(args, seed, variants)

    gain = variants["full_step10"]["prediction_vs_mean_with_identical_step10_head"]
    print(
        f"seed={seed} "
        f"angle Gate={angles['gate_latent_vs_mean']['mean_angle_degrees']:.3f}° "
        f"Projection={angles['projection_vs_mean']['mean_angle_degrees']:.3f}° "
        f"Both={angles['full_vs_mean']['mean_angle_degrees']:.3f}° | "
        f"val ΔMSE Full-vs-MeanHead={gain['mean_squared_error_delta']:+.6f}",
        flush=True)
    report = {
        "seed": seed,
        "reference_mean_checkpoint": str(reference),
        "exact_E0_step0_1_10_replay": checks,
        "initial_max_pool_vs_mean_error": identity_err,
        "pool_group_sizes": {
            "gate_latent_state_tensors": len(gate_keys),
            "projection_state_tensors": len(projection_keys),
        },
        "full_raw_equals_projection_of_gate_raw_max_abs_error": algebra_err,
        "embedding_angular_drift": angles,
        "counterfactuals": variants,
    }
    del model, optimizer, train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def mean_std(x):
    a = np.asarray(x, dtype=np.float64)
    return {"mean": float(a.mean()), "std_across_seeds": float(a.std(ddof=1)),
            "per_seed": {str(seed): float(value)
                         for seed, value in zip(SEEDS, a.tolist())}}


def aggregate(report):
    rows = report["seeds"]
    summary = {"n_seeds": len(SEEDS), "seed_ids": list(SEEDS)}
    for contrast in ANGLE_CONTRASTS:
        summary[f"angle_{contrast}_degrees"] = mean_std([
            rows[str(seed)]["embedding_angular_drift"][contrast][
                "mean_angle_degrees"] for seed in SEEDS])
    for name in ("gate_latent_only", "projection_only", "full_step10"):
        summary[f"{name}_cosine_drift_vs_mean"] = mean_std([
            rows[str(seed)]["counterfactuals"][name][
                "pair_similarity_drift_vs_mean"]["mean_abs_cosine_delta"]
            for seed in SEEDS])
        summary[f"{name}_validation_delta_spearman_vs_mean"] = mean_std([
            rows[str(seed)]["counterfactuals"][name]["validation"]["spearman"]
            - rows[str(seed)]["counterfactuals"]["mean"]["validation"]["spearman"]
            for seed in SEEDS])
        summary[f"{name}_validation_delta_mse_vs_same_head_mean"] = mean_std([
            rows[str(seed)]["counterfactuals"][name][
                "prediction_vs_mean_with_identical_step10_head"][
                    "mean_squared_error_delta"]
            for seed in SEEDS])
        summary[f"{name}_fraction_pairs_MSE_improved"] = mean_std([
            rows[str(seed)]["counterfactuals"][name][
                "prediction_vs_mean_with_identical_step10_head"][
                    "fraction_pairs_squared_error_improved"]
            for seed in SEEDS])
    summary["num_seeds_gate_angular_drift_exceeds_projection"] = sum(
        rows[str(seed)]["embedding_angular_drift"]["gate_latent_vs_mean"][
            "mean_angle_degrees"] > rows[str(seed)]["embedding_angular_drift"][
                "projection_vs_mean"]["mean_angle_degrees"]
        for seed in SEEDS)
    summary["num_seeds_gate_cosine_drift_exceeds_projection"] = sum(
        rows[str(seed)]["counterfactuals"]["gate_latent_only"][
            "pair_similarity_drift_vs_mean"]["mean_abs_cosine_delta"] >
        rows[str(seed)]["counterfactuals"]["projection_only"][
            "pair_similarity_drift_vs_mean"]["mean_abs_cosine_delta"]
        for seed in SEEDS)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model_path",
        default="/home/data/home/wwr_lumos/models/roberta-base")
    ap.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    ap.add_argument("--pilot_dir",
                    default="outputs/text/stsb_frozen_trained_mean")
    ap.add_argument("--e3b_root",
                    default="outputs/text/stsb_mean_equiv_e3b_step10")
    ap.add_argument(
        "--output",
        default="outputs/text/stsb_mean_equiv_e4_representation_drift/"
                "e4_five_seed_step10_representation_drift.json")
    ap.add_argument("--check-only", action="store_true",
                    help="Validate five E0 source sets, do not train or save.")
    args = ap.parse_args()
    out = Path(args.output)
    if out.exists() and not args.check_only:
        raise FileExistsError(
            f"Results already exist, no overwrite: {out}")
    sources = preflight(args)
    print("E4 five-seed representation drift preflight PASSED: 0,1,2,3,4",
          flush=True)
    if args.check_only:
        print("Check-only complete; NO output and NO training.", flush=True)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True)
    print("Training = EXACT replay of first 10 E0 steps PER SEED. "
          "No new model training or E0 modifications.", flush=True)
    report = {
        "experiment": "E4_exact_E0_step10_five_seed_representation_drift",
        "seeds_requested": list(SEEDS),
        "n_validation_pairs_per_seed": 1500,
        "n_validation_sentence_occurrences_per_seed": 3000,
        "interpretation": {
            "mean": "Frozen trained RoBERTa masked Mean, Step10 Head.",
            "gate_latent_only": "Exact Step10 Gate+Latent output BEFORE "
                                "Projection, normalized, Step10 Head.",
            "projection_only": "Exact Step10 Projection applied to "
                               "original frozen-encoder Mean, Step10 Head.",
            "full_step10": "Exact jointly trained Step10 B2 output, "
                           "Step10 Head.",
            "embedding_angle": "Angle between corresponding L2-normalized "
                               "sentence representations; 3000 sentence "
                               "OCCURRENCES from 1500 STSB validation pairs, "
                               "not 3000 necessarily distinct sentences.",
            "MSE_help": "Per-pair squared-error comparison at SAME Step10 "
                        "positive-affine Head vs unmodified Mean.",
            "causality": "Inference-time path intervention at one joint "
                         "optimizer checkpoint, NOT separately trained "
                         "Gate-only/Projection-only models.",
        },
        "seeds": {},
    }
    for seed in SEEDS:
        report["seeds"][str(seed)] = replay_seed(
            args, seed, tokenizer, sources, device)
    report["summary"] = aggregate(report)
    report["limitations"] = [
        "Representation angular drift and performance damage are different "
        "quantities; large angular change does not imply large Spearman drop.",
        "Gate+Latent includes frequency Attention, FFN and Gate parameters "
        "updated JOINTLY with Projection/Head.",
        "Post-pool projection may improve or worsen MSE depending on the "
        "input state; all module effects are CONDITIONAL and not additive.",
        "The original Mean was selected on the same validation split. "
        "No independent held-out test conclusion.",
        "Only ten Adam updates, frozen RoBERTa and STSB. Does not establish "
        "why some other tasks benefit, or the full-epoch mechanism.",
    ]
    content = json.dumps(
        report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        f.write(content)
    print("\n=== Five-seed summary ===", flush=True)
    for contrast in ANGLE_CONTRASTS:
        s = report["summary"][f"angle_{contrast}_degrees"]
        print(f"{contrast:<25} angle={s['mean']:.3f} ± "
              f"{s['std_across_seeds']:.3f} degrees", flush=True)
    for name in ("gate_latent_only", "projection_only", "full_step10"):
        s = report["summary"][f"{name}_validation_delta_spearman_vs_mean"]
        m = report["summary"][f"{name}_validation_delta_mse_vs_same_head_mean"]
        print(f"{name:<25} ΔSpearman={s['mean']:+.6f} "
              f"ΔMSE={m['mean']:+.6f}", flush=True)
    print(f"Saved exactly ONE fixed result: {out}", flush=True)


if __name__ == "__main__":
    main()
