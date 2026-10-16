#!/usr/bin/env python3
"""E3c: Step-10 real/imag Gate symmetry and padding mechanism diagnostics.

Replay E0's first TEN joint Adam steps, prove exact Step0/1/10 validation
predictions, then run inference-only counterfactual forwards at Step10:
  original_gate: learned independent real/imag multipliers a,b
  tied_real_gate: force imaginary multiplier b=a (DC/real gate unchanged)
  identity_gate: force every multiplier=1 (retains learned Projection)
  mean_current_head: original frozen-encoder masked Mean + Step10 score head
  padded_fixed_gate: add 32 zero-hidden tokens, REUSE unpadded original logits
  padded_recomputed_gate: add 32 zero-hidden tokens, recompute learned logits
  padded_tied_fixed_gate: tied-real gate with same unpadded real gate
The same original valid-token encoder outputs are reused within each batch,
so padding tests do NOT alter RoBERTa representations.

No new model is trained beyond exact 10-step replay. No original outputs are
modified. Writes one fixed, exclusive JSON result only after full checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata, spearmanr
from transformers import AutoTokenizer

from e2_adam_step1_counterfactual import compare_with_e0
from e3b_replay_step10_module_splices import (
    load_e0_metadata, numeric_check,
)
from explore_mean_equiv_trajectory import evaluate, get_loaders, get_splits
from train_author_text_protocol import seed_everything
from train_stsb_frozen_mean_start import (
    assert_initial_pool_identity,
    build_frozen_b2,
)
from AMPCliff.factory.pooling.llm_pooling_dropin import masked_mean_pooling


VARIANTS = (
    "original_gate",
    "tied_real_gate",
    "identity_gate",
    "mean_current_head",
    "padded_fixed_gate",
    "padded_recomputed_gate",
    "padded_tied_fixed_gate",
)
PAIRS = ("a", "b")


def compute_metrics(labels, score):
    y = np.asarray(labels, dtype=np.float64)
    pred = np.asarray(score, dtype=np.float64)
    return {
        "mse": float(np.mean((pred - y / 5.0) ** 2)),
        "spearman": float(spearmanr(y, pred).statistic),
        "pearson": float(pearsonr(y, pred).statistic),
    }


def pool_with_gate_intervention(
        model, hidden, mask, mode, fixed_logits=None):
    """Call the EXACT learned Pooling while intervening at final Gate logits.

    Pair real/imag indices by channel. Since g=2*sigmoid(logits), copying
    real logits into imaginary channels enforces b=a identically.
    No parameters are edited; forward hook is removed even on exceptions.
    """
    d_model = hidden.shape[-1]
    seen = []

    def hook(_module, _input, logits):
        if logits.ndim != 2 or logits.shape[-1] != 2 * d_model:
            raise RuntimeError("Unexpected Gate-logit shape.")
        if fixed_logits is not None:
            if fixed_logits.shape != logits.shape:
                raise RuntimeError("Cannot reuse logits: batch shape changed.")
            current = fixed_logits.to(device=logits.device, dtype=logits.dtype)
        else:
            current = logits
        seen.append(current.detach().float().cpu())
        if mode in ("tied_real_gate", "padded_tied_fixed_gate"):
            return torch.cat((current[:, :d_model],
                              current[:, :d_model]), dim=-1)
        if mode == "identity_gate":
            return torch.zeros_like(current)
        if mode in ("original_gate", "padded_fixed_gate",
                    "padded_recomputed_gate"):
            return current
        raise ValueError(f"Unknown Gate mode: {mode}")

    handle = model.pool.freq_gate[-1].register_forward_hook(hook)
    try:
        pooled = model.pool(hidden, attention_mask=mask)
    finally:
        handle.remove()
    if len(seen) != 1:
        raise RuntimeError("Expected exactly one final Gate invocation.")
    return F.normalize(pooled, p=2, dim=-1), seen[0]


def score_from_embeddings(model, x, y):
    cosine = (x * y).sum(dim=-1)
    scores = (model.log_scale.exp() * cosine + model.bias)
    return (cosine.detach().cpu().numpy().astype(np.float32),
            scores.detach().cpu().numpy().astype(np.float32))


def evaluate_counterfactuals(model, val_loader, extra_pad_tokens):
    model.eval()
    collected = {
        key: {"cos": [], "score": []} for key in VARIANTS
    }
    label_parts = []
    gate_stats = {
        "n_real_imag_channel_pairs": 0, "sum_abs_difference": 0.,
        "max_abs_difference": 0., "count_abs_gt_0_1": 0,
        "sum_abs_real_from_1": 0., "sum_abs_imag_from_1": 0.,
    }
    # Note: all synthetic padding is applied AFTER the frozen RoBERTa
    # forward, using exactly the same valid hidden states.
    with torch.no_grad():
        for token_a, token_b, labels in val_loader:
            hidden_sides = []
            mask_sides = []
            for tokens in (token_a, token_b):
                tokens = {key: value.to(next(model.parameters()).device)
                          for key, value in tokens.items()}
                hidden = model.encoder(**tokens).last_hidden_state
                mask = tokens["attention_mask"]
                hidden_sides.append(hidden)
                mask_sides.append(mask)
            label_parts.append(labels.numpy().astype(np.float32))

            embeddings = {key: [] for key in VARIANTS}
            for hidden, mask in zip(hidden_sides, mask_sides):
                baseline_emb, base_logits = pool_with_gate_intervention(
                    model, hidden, mask, "original_gate")
                embeddings["original_gate"].append(baseline_emb)
                d = hidden.shape[-1]
                gate_real = 2.0 * torch.sigmoid(base_logits[:, :d])
                gate_imag = 2.0 * torch.sigmoid(base_logits[:, d:])
                diff = (gate_real - gate_imag).abs()
                gate_stats["n_real_imag_channel_pairs"] += int(diff.numel())
                gate_stats["sum_abs_difference"] += float(diff.sum())
                gate_stats["max_abs_difference"] = max(
                    gate_stats["max_abs_difference"], float(diff.max()))
                gate_stats["count_abs_gt_0_1"] += int((diff > 0.1).sum())
                gate_stats["sum_abs_real_from_1"] += float(
                    (gate_real - 1.0).abs().sum())
                gate_stats["sum_abs_imag_from_1"] += float(
                    (gate_imag - 1.0).abs().sum())

                tied, _ = pool_with_gate_intervention(
                    model, hidden, mask, "tied_real_gate",
                    fixed_logits=base_logits)
                embeddings["tied_real_gate"].append(tied)
                identity, _ = pool_with_gate_intervention(
                    model, hidden, mask, "identity_gate")
                embeddings["identity_gate"].append(identity)
                mean = F.normalize(masked_mean_pooling(
                    hidden, mask, eps=1e-6), p=2, dim=-1)
                embeddings["mean_current_head"].append(mean)

                hidden_padded = F.pad(
                    hidden, (0, 0, 0, extra_pad_tokens), value=0.)
                mask_padded = F.pad(
                    mask, (0, extra_pad_tokens), value=0)
                fixed, _ = pool_with_gate_intervention(
                    model, hidden_padded, mask_padded,
                    "padded_fixed_gate", fixed_logits=base_logits)
                recomputed, _ = pool_with_gate_intervention(
                    model, hidden_padded, mask_padded,
                    "padded_recomputed_gate")
                tied_padded, _ = pool_with_gate_intervention(
                    model, hidden_padded, mask_padded,
                    "padded_tied_fixed_gate", fixed_logits=base_logits)
                embeddings["padded_fixed_gate"].append(fixed)
                embeddings["padded_recomputed_gate"].append(recomputed)
                embeddings["padded_tied_fixed_gate"].append(tied_padded)

            for key in VARIANTS:
                cosine, scores = score_from_embeddings(
                    model, embeddings[key][0], embeddings[key][1])
                collected[key]["cos"].append(cosine)
                collected[key]["score"].append(scores)
    labels = np.concatenate(label_parts)
    for row in collected.values():
        for key in ("cos", "score"):
            row[key] = np.concatenate(row[key])
            if row[key].shape != labels.shape:
                raise RuntimeError("Validation prediction shape mismatch.")
    if not np.isfinite(labels).all():
        raise RuntimeError("Nonfinite labels.")
    for name, row in collected.items():
        if not np.isfinite(row["cos"]).all() or not np.isfinite(row["score"]).all():
            raise RuntimeError(f"Nonfinite predictions in {name}.")
    n = gate_stats["n_real_imag_channel_pairs"]
    gate_summary = {
        "n_sentence_channel_pairs": n,
        "mean_abs_a_minus_b": gate_stats["sum_abs_difference"] / n,
        "max_abs_a_minus_b": gate_stats["max_abs_difference"],
        "fraction_abs_a_minus_b_gt_0_1":
            gate_stats["count_abs_gt_0_1"] / n,
        "mean_abs_a_minus_1": gate_stats["sum_abs_real_from_1"] / n,
        "mean_abs_b_minus_1": gate_stats["sum_abs_imag_from_1"] / n,
        "a_b_range": "a,b each in (0,2), a=real, b=imag; "
                     "shared across all FFT bins per sentence/channel",
    }
    return labels, collected, gate_summary


def compare_npz_with_e0(labels, row, path, threshold=2e-4):
    arrays = {"labels": labels, "score_full": row["score"],
              "cos_full": row["cos"]}
    return compare_with_e0(arrays, path, "E0 Step10 (E3c cached-hidden)",
                           threshold=threshold)


def summarize_variants(labels, rows):
    reference = rows["original_gate"]
    baseline_cos = reference["cos"].astype(np.float64)
    baseline_ranks = rankdata(baseline_cos)
    results = {}
    for name, row in rows.items():
        cos = row["cos"].astype(np.float64)
        ranks = rankdata(cos)
        results[name] = {
            "validation": compute_metrics(labels, row["score"]),
            "relative_to_original_gate": {
                "mean_abs_pair_cos_change":
                    float(np.mean(np.abs(cos - baseline_cos))),
                "max_abs_pair_cos_change":
                    float(np.max(np.abs(cos - baseline_cos))),
                "mean_abs_rank_change":
                    float(np.abs(ranks - baseline_ranks).mean()),
                "fraction_pairs_moved_at_least_50_ranks":
                    float((np.abs(ranks - baseline_ranks) >= 50).mean()),
            },
        }
    for row in results.values():
        row["delta_validation_vs_original_gate"] = {
            metric: row["validation"][metric] -
            results["original_gate"]["validation"][metric]
            for metric in ("mse", "spearman", "pearson")
        }
    return results


def no_overwrite(path, report):
    if path.exists():
        raise FileExistsError(f"E3c result exists: {path}; no overwrite.")
    content = json.dumps(report, ensure_ascii=False, indent=2,
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
                        default="outputs/text/stsb_mean_equiv_e3c_gate_symmetry")
    parser.add_argument("--extra_pad_tokens", type=int, default=32)
    parser.add_argument("--check-only", action="store_true",
                        help="Verify original input and Step0 only; save nothing.")
    args = parser.parse_args()
    if args.seed < 0 or args.extra_pad_tokens < 1:
        parser.error("--seed must be >=0; --extra_pad_tokens must be >=1.")
    source = Path(args.pilot_dir) / f"seed_{args.seed}"
    output = (Path(args.output_root) / f"seed_{args.seed}" /
              "e3c_step10_gate_symmetry.json")
    if output.exists() and not args.check_only:
        raise FileExistsError(f"E3c result already exists: {output}")

    summary_path, summary, e0, checkpoint = load_e0_metadata(
        source, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"E3c seed={args.seed}, device={device}, extra_pad="
          f"{args.extra_pad_tokens} hidden tokens", flush=True)
    print("Same trained Mean start; replay only 10 E0 updates.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True)
    train_data, val_data = get_splits(
        "stsb", args.stsb_data_path, "", args.seed, 0, 0)
    if (len(train_data) != int(summary["n_train"])
            or len(val_data) != int(summary["n_validation"])):
        raise RuntimeError("E0 dataset cardinalities changed.")
    train_loader, val_loader = get_loaders(
        "stsb", tokenizer, train_data, val_data, args.seed)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_frozen_b2(args, device, state)
    del state
    identity_difference = assert_initial_pool_identity(
        model, val_loader, device)
    model.eval()
    initial_eval, initial_pairs = evaluate(
        model, val_loader, "stsb", device)
    checks = {"step0": compare_with_e0(
        initial_pairs, source / "validation_pairs_step_0.npz",
        "E0 Step0")}
    numeric_check(initial_eval["full"], e0[0], 0)
    if args.check_only:
        print("CHECK-ONLY: E0 input and Step0 reproduced; NO output.",
              flush=True)
        return

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=0.)
    seed_everything(args.seed + 100000)
    model.train()
    online_losses = []
    for step, (tok_a, tok_b, label) in enumerate(train_loader, start=1):
        if step > 10:
            break
        a = {k: v.to(device) for k, v in tok_a.items()}
        b = {k: v.to(device) for k, v in tok_b.items()}
        optimizer.zero_grad(set_to_none=True)
        yhat, _ = model(a, b)
        loss = F.mse_loss(yhat, label.to(device) / 5.)
        loss.backward()
        optimizer.step()
        online_losses.append(float(loss.detach().cpu()))
        if step in (1, 10):
            model.eval()
            result, arrays = evaluate(model, val_loader, "stsb", device)
            checks[f"step{step}"] = compare_with_e0(
                arrays, source / f"validation_pairs_step_{step}.npz",
                f"E0 Step{step}")
            numeric_check(result["full"], e0[step], step)
            model.train()
    if len(online_losses) != 10:
        raise RuntimeError("Only fewer than 10 training batches.")
    for step, index in ((1, 0), (10, 9)):
        expected = float(e0[step]["last_minibatch_train_mse"])
        if abs(online_losses[index] - expected) > 1e-7:
            raise RuntimeError(f"E0 Step{step} minibatch loss mismatch.")

    print("Step 0/1/10 E0 reproduction passed; evaluating Gate interventions.",
          flush=True)
    model.eval()
    labels, pairs, gate_summary = evaluate_counterfactuals(
        model, val_loader, args.extra_pad_tokens)
    check_cached = compare_npz_with_e0(
        labels, pairs["original_gate"],
        source / "validation_pairs_step_10.npz")
    variants = summarize_variants(labels, pairs)
    numeric_check(variants["original_gate"]["validation"], e0[10], 10)
    # Strong structural test: tied-real gate scales each time token/channel
    # and thus its masked mean must be invariant to right zero padding when
    # the GATE logits and original hidden states are fixed.
    original_tied = pairs["tied_real_gate"]["cos"]
    padded_tied = pairs["padded_tied_fixed_gate"]["cos"]
    tied_pad_max = float(np.max(np.abs(
        original_tied.astype(np.float64) -
        padded_tied.astype(np.float64))))
    if tied_pad_max > 2e-4:
        raise RuntimeError(
            f"Symmetric fixed Gate unexpected padding dependence: "
            f"max pair cosine diff={tied_pad_max:.3e}")
    if len(labels) != int(summary["n_validation"]):
        raise RuntimeError("Validation pair count differs from E0.")

    report = {
        "experiment": "E3c_step10_gate_real_imag_symmetry_padding",
        "seed": args.seed,
        "frozen_encoder": True,
        "trained_mean_start": str(checkpoint),
        "original_e0_summary": str(summary_path),
        "exact_e0_replay": checks,
        "cached_hidden_e0_step10_prediction_check": check_cached,
        "model_protocol": "Frozen trained RoBERTa, E0 joint B2 10-step Adam; "
                          "time masked Mean, Projection initial identity, "
                          "Step10 Head held fixed for all counterfactuals",
        "extra_right_zero_hidden_tokens": args.extra_pad_tokens,
        "padding_protocol": "Same already-encoded hidden states; append "
                            "zero hidden features and zero attention mask "
                            "AFTER frozen RoBERTa; no second encoder pass",
        "a_b_gate_statistics": gate_summary,
        "variants": variants,
        "tied_real_padded_fixed_cos_max_abs_difference": tied_pad_max,
        "interpretive_notes": [
            "tied_real_gate forces b=a by replacing imaginary-channel "
            "Gate logits with real-channel logits. It is NOT necessarily "
            "Mean pooling: real/imag equal still permits sentence-wise "
            "and feature-wise reweighting, and Projection remains learned.",
            "identity_gate forces a=b=1 while retaining Step10 Projection; "
            "it is the Step10 Projection-only counterfactual.",
            "mean_current_head bypasses the entire B2 pool and projection; "
            "it preserves original trained Mean cosine rankings.",
            "padded_fixed_gate isolates FFT/iFFT/mean's sensitivity to "
            "appended right-padding at identical a,b and identical input "
            "hidden tokens; padded_recomputed_gate also changes Gate "
            "because FFT length and attention bins differ.",
            "This intervention studies sufficiency of a structural "
            "asymmetry at an existing jointly trained checkpoint; it "
            "does not identify the independent training trajectory.",
            "The original Mean reference was validation-selected; "
            "this is NOT an independent test result.",
            "Even if tying a=b improves validation Spearman, this alone "
            "does not prove the optimizer should learn equality: "
            "the training objective is MSE, not validation Spearman.",
        ],
    }
    print("Gate mean|a-b|:",
          f"{gate_summary['mean_abs_a_minus_b']:.6f}", flush=True)
    print(f"{'Variant':<30} {'Val Spearman':>13} {'ΔSpearman':>12} "
          f"{'Val MSE':>11} {'Mean |Δcos|':>13}", flush=True)
    for name, row in variants.items():
        print(f"{name:<30} "
              f"{row['validation']['spearman']:>13.6f} "
              f"{row['delta_validation_vs_original_gate']['spearman']:>+12.6f} "
              f"{row['validation']['mse']:>11.7f} "
              f"{row['relative_to_original_gate']['mean_abs_pair_cos_change']:>13.6f}",
              flush=True)
    print("Symmetric Gate fixed-logits pad-invariance max cosine diff:",
          f"{tied_pad_max:.3e}", flush=True)
    no_overwrite(output, report)
    print("Saved E3c JSON (fixed, no overwrite):", output, flush=True)


if __name__ == "__main__":
    main()
