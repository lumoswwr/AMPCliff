#!/usr/bin/env python3
"""E6: five-seed *decision experiment* for overshoot vs harmful direction.

From EXACT trained-Mean-equivalent E0 Step10 models, WITHOUT refitting,
compare smaller departures from Mean via:
  (1) scalar shrinkage of GP *predictions*, including analytic train MSE
      optimum and its independent evaluation on the validation split;
  (2) functional Gate/Projection strength interpolation on a declared
      5x5 (alpha,beta) grid and 0.125 additional joint-path midpoint.

Train-selected grid settings are assessed on validation without choosing
from validation. Critically distinguish:
  recover from full-model damage (less bad than GP at alpha=beta=1)
  vs genuine benefit over Mean (better than alpha=beta=0).

Five seeds (0-4) in ONE invocation. Exactly TEN original E0 optimizer
steps per seed for checkpoint recovery, no new training, one exclusive JSON.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from e2_adam_step1_counterfactual import compare_with_e0
from e3b_replay_step10_module_splices import numeric_check
from e4_five_seed_representation_drift import SEEDS, preflight
from explore_mean_equiv_trajectory import (
    evaluate, get_loaders, get_splits, metrics,
)
from train_author_text_protocol import PairCollator, PairDataset, seed_everything
from train_stsb_frozen_mean_start import (
    assert_initial_pool_identity, build_frozen_b2,
)
from AMPCliff.factory.pooling.llm_pooling_dropin import masked_mean_pooling


STRENGTHS = (0., .25, .5, .75, 1.)
DIAG_STRENGTHS = (0., .125, .25, .5, .75, 1.)
SCORE_STRENGTHS = (0., .125, .25, .5, .75, 1.)
GRID = tuple((a, b) for a in STRENGTHS for b in STRENGTHS)
PATH = tuple((lam, lam) for lam in DIAG_STRENGTHS)
POINTS = tuple(dict.fromkeys(GRID + PATH))


def point_id(a, b):
    return f"a{a:.3f}_b{b:.3f}"


MEAN_ID = point_id(0., 0.)
FULL_ID = point_id(1., 1.)
GATE_ID = point_id(1., 0.)
PROJ_ID = point_id(0., 1.)


def evaluate_path_grid(model, loader, device):
    """One RoBERTa pass per sentence, all counterfactuals from same hidden.

    Let raw Mean m, raw trained Gate g, trained Projection P(z)=Wz+b.
    z_alpha=(1-alpha)m+alpha*g
    z_alpha_beta=(1-beta)z_alpha+beta*P(z_alpha).
    Because P is affine, P(z_alpha)=(1-alpha)P(m)+alpha*P(g).
    L2 normalize *after* interpolation, then use same Step10 score Head.
    """
    model.eval()
    labels_chunks = []
    score_chunks = {point_id(*pt): [] for pt in POINTS}
    full_cos_chunks = []
    max_composition_error = 0.
    with torch.no_grad():
        for toks_a, toks_b, y in loader:
            labels_chunks.append(y.numpy().astype(np.float32))
            sides = []
            for tokens in (toks_a, toks_b):
                tokens = {k: v.to(device) for k, v in tokens.items()}
                mask = tokens["attention_mask"]
                hidden = model.encoder(**tokens).last_hidden_state
                m = masked_mean_pooling(hidden, mask, eps=1e-6)
                full = model.pool(hidden, attention_mask=mask)
                g = model.pool._last_pooled_pre_projection
                pm = model.pool.time_out_proj(m)
                pg = model.pool.time_out_proj(g)
                max_composition_error = max(
                    max_composition_error, float((full-pg).abs().max()))
                if max_composition_error > 2e-6:
                    raise RuntimeError(
                        "Projection(Gate preprojection) != actual FLaG_B2. "
                        "Check dropout/postnorm protocol.")
                side = {}
                for a in tuple(dict.fromkeys(x[0] for x in POINTS)):
                    z = m + a * (g-m)
                    pz = pm + a * (pg-pm)
                    for b in tuple(dict.fromkeys(x[1] for x in POINTS
                                                 if x[0] == a)):
                        key = point_id(a, b)
                        raw = z + b * (pz-z)
                        side[key] = F.normalize(raw, p=2, dim=-1)
                sides.append(side)
            for pt in POINTS:
                key = point_id(*pt)
                cos = (sides[0][key] * sides[1][key]).sum(dim=-1)
                score = model.log_scale.exp() * cos + model.bias
                score_chunks[key].append(
                    score.detach().cpu().numpy().astype(np.float32))
                if key == FULL_ID:
                    full_cos_chunks.append(
                        cos.detach().cpu().numpy().astype(np.float32))
    labels = np.concatenate(labels_chunks)
    scores = {k: np.concatenate(v).astype(np.float64)
              for k, v in score_chunks.items()}
    cosine_full = np.concatenate(full_cos_chunks)
    if any(v.shape != labels.shape for v in scores.values()):
        raise RuntimeError("Interpolation prediction size/order changed.")
    if any(not np.isfinite(v).all() for v in scores.values()):
        raise RuntimeError("Nonfinite interpolation prediction.")
    return labels, scores, cosine_full, max_composition_error


def mse(labels, values):
    d = np.asarray(values, dtype=np.float64) - (
        np.asarray(labels, dtype=np.float64) / 5.)
    return float(np.mean(d*d))


def score_stats(labels, pred):
    result = metrics("stsb", labels, pred)
    if not all(np.isfinite(result[k]) for k in ("mse","spearman","pearson")):
        raise RuntimeError("Nonfinite STSB metric.")
    return result


def best_train_point(train_rows, field, candidates):
    """Train-only selection, deterministic tie-break toward smaller change."""
    return min(candidates, key=lambda pair: (
        -train_rows[point_id(*pair)][field] if field == "spearman"
        else train_rows[point_id(*pair)][field],
        pair[0]+pair[1], pair[0], pair[1]))


def loss_optimal_prediction_shrinkage(labels, m, gp):
    """No grid search: analytic optimal lambda of M + lambda*(GP-M)."""
    y = np.asarray(labels, dtype=np.float64) / 5.
    delta = gp-m
    v = float(np.mean(delta*delta))
    if not np.isfinite(v) or v <= 1e-20:
        raise RuntimeError("Degenerate GP-M score trajectory.")
    grad0 = 2 * float(np.mean((m-y)*delta))
    optimum = -grad0 / (2*v)
    return {
        "optimum_unconstrained": optimum,
        "optimum_clipped_0_to_1": float(np.clip(optimum, 0., 1.)),
        "mse_derivative_at_mean_lambda0": grad0,
        "mse_derivative_at_full_lambda1": grad0+2*v,
        "curvature_2_mean_delta_squared": 2*v,
        "interpretation": (
            "Derivative at lambda=0 >= 0 means the GP-M prediction "
            "direction cannot improve MSE for infinitesimal positive "
            "steps on this split. A clipped optimum inside (0,1) "
            "means finite-amplitude overshoot along this fixed "
            "prediction direction, not proven parameter-optimizer overshoot."),
    }


def analyze_split(labels, scores):
    rows = {key: score_stats(labels, pred)
            for key, pred in scores.items()}
    score_path = {
        f"lambda_{lam:.3f}": score_stats(
            labels, scores[MEAN_ID]+lam*(scores[FULL_ID]-scores[MEAN_ID]))
        for lam in SCORE_STRENGTHS
    }
    return {
        "n_pairs": len(labels),
        "grid_scores": rows,
        "prediction_linear_shrinkage": {
            "analytic_MSE_direction":
                loss_optimal_prediction_shrinkage(
                    labels, scores[MEAN_ID], scores[FULL_ID]),
            "fixed_lambda_metrics": score_path,
        },
    }


def check_e4(path, seed, val_rows):
    old = json.loads(Path(path).read_text(encoding="utf-8"))
    mappings = {
        "mean": MEAN_ID,
        "gate_latent_only": GATE_ID,
        "projection_only": PROJ_ID,
        "full_step10": FULL_ID,
    }
    largest = 0.
    for old_name, new_name in mappings.items():
        for metric in ("mse", "spearman", "pearson"):
            diff = abs(float(old["seeds"][str(seed)]["counterfactuals"][
                old_name]["validation"][metric]) -
                       float(val_rows[new_name][metric]))
            largest = max(largest, diff)
            if diff > 5e-6:
                raise RuntimeError(
                    f"E6 disagree E4 seed{seed} {old_name}/{metric} "
                    f"difference={diff:.6g}")
    return {"status": "passed", "max_abs_metric_diff": largest}


def selection_comparison(train_analysis, val_analysis, train_scores, val_scores,
                         train_labels, val_labels):
    tr = train_analysis["grid_scores"]
    vl = val_analysis["grid_scores"]
    selections = {}
    requested = {
        "train_MSE_best_joint_path":
            best_train_point(tr, "mse", PATH),
        "train_MSE_best_two_dim_grid":
            best_train_point(tr, "mse", GRID),
        "train_Spearman_best_joint_path":
            best_train_point(tr, "spearman", PATH),
        "train_Spearman_best_two_dim_grid":
            best_train_point(tr, "spearman", GRID),
    }
    for desc, pt in requested.items():
        key = point_id(*pt)
        selections[desc] = {
            "alpha": pt[0], "beta": pt[1],
            "selected_using": "TRAIN ONLY",
            "train": tr[key],
            "validation": vl[key],
            "validation_delta_MSE_vs_mean":
                vl[key]["mse"]-vl[MEAN_ID]["mse"],
            "validation_delta_MSE_vs_original_full":
                vl[key]["mse"]-vl[FULL_ID]["mse"],
            "validation_delta_Spearman_vs_mean":
                vl[key]["spearman"]-vl[MEAN_ID]["spearman"],
            "validation_delta_Spearman_vs_original_full":
                vl[key]["spearman"]-vl[FULL_ID]["spearman"],
        }

    # Analytic score-space optimum is computed on TRAIN *only*.
    lstar = train_analysis["prediction_linear_shrinkage"][
        "analytic_MSE_direction"]["optimum_clipped_0_to_1"]
    train_m = train_scores[MEAN_ID]
    train_gp = train_scores[FULL_ID]
    val_m = val_scores[MEAN_ID]
    val_gp = val_scores[FULL_ID]
    tr_metric = score_stats(train_labels, train_m+lstar*(train_gp-train_m))
    va_metric = score_stats(val_labels, val_m+lstar*(val_gp-val_m))
    selections["train_MSE_analytic_prediction_shrinkage"] = {
        "lambda": lstar,
        "selected_using": "TRAIN ONLY, analytic minimum",
        "train": tr_metric,
        "validation": va_metric,
        "validation_delta_MSE_vs_mean":
            va_metric["mse"]-vl[MEAN_ID]["mse"],
        "validation_delta_MSE_vs_original_full":
            va_metric["mse"]-vl[FULL_ID]["mse"],
        "validation_delta_Spearman_vs_mean":
            va_metric["spearman"]-vl[MEAN_ID]["spearman"],
        "validation_delta_Spearman_vs_original_full":
            va_metric["spearman"]-vl[FULL_ID]["spearman"],
    }
    return selections


def run_seed(args, seed, tokenizer, source, device):
    pilot, _, summary, probes, reference = source
    args.seed = seed
    print(f"\n[E6 seed {seed}] exact E0 10-step reproduction", flush=True)
    train, val = get_splits(
        "stsb", args.stsb_data_path, "", seed, 0, 0)
    if len(train) != summary["n_train"] or len(val) != summary["n_validation"]:
        raise RuntimeError(f"Seed {seed}: STSB splits changed.")
    online_loader, val_loader = get_loaders(
        "stsb", tokenizer, train, val, seed)
    mean_state = torch.load(
        reference, map_location="cpu", weights_only=True)
    if any(k.startswith("pool.") for k in mean_state):
        raise RuntimeError("A trained Mean-only state_dict is required.")
    model = build_frozen_b2(args, device, mean_state)
    del mean_state
    identity_error = assert_initial_pool_identity(
        model, val_loader, device)

    ev0, arr0 = evaluate(model, val_loader, "stsb", device)
    replay = {
        "step0": compare_with_e0(
            arr0, pilot/"validation_pairs_step_0.npz",
            f"E0 Step0 seed{seed}"),
    }
    numeric_check(ev0["full"], probes[0], 0)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=0.)
    seed_everything(seed + 100000)
    model.train()
    online_losses = []
    for step, (toks_a, toks_b, y) in enumerate(online_loader, 1):
        if step > 10:
            break
        toks_a = {k: v.to(device) for k, v in toks_a.items()}
        toks_b = {k: v.to(device) for k, v in toks_b.items()}
        optimizer.zero_grad(set_to_none=True)
        pred, _ = model(toks_a, toks_b)
        loss = F.mse_loss(pred, y.to(device)/5.)
        loss.backward()
        optimizer.step()
        online_losses.append(float(loss.detach().cpu()))
        if step in (1, 10):
            ev, arr = evaluate(model, val_loader, "stsb", device)
            replay[f"step{step}"] = compare_with_e0(
                arr, pilot/f"validation_pairs_step_{step}.npz",
                f"E0 Step{step} seed{seed}")
            numeric_check(ev["full"], probes[step], step)
            model.train()
    if len(online_losses) != 10:
        raise RuntimeError("Expected exactly ten E0 minibatches.")
    for step, i in ((1,0),(10,9)):
        if abs(online_losses[i]-
               float(probes[step]["last_minibatch_train_mse"]))>1e-7:
            raise RuntimeError(
                f"E0 online Step{step} loss mismatch, seed={seed}")

    # Both splits EVALUATED at batch size 64; original E0 replay retains
    # its batch size 32 for TRAIN optimizer and 64 for validation checks.
    eval_collator = PairCollator(
        tokenizer, max_length=128, label_dtype=torch.float32)
    train_eval_loader = DataLoader(
        PairDataset(train, "stsb"), batch_size=64,
        shuffle=False, collate_fn=eval_collator, num_workers=0)
    labels_train, scores_train, _, error_train = evaluate_path_grid(
        model, train_eval_loader, device)
    labels_val, scores_val, cos_val, error_val = evaluate_path_grid(
        model, val_loader, device)

    replay["step10_interpolated_full"] = compare_with_e0({
        "labels": labels_val,
        "score_full": scores_val[FULL_ID].astype(np.float32),
        "cos_full": cos_val,
    }, pilot/"validation_pairs_step_10.npz",
        f"E0 Step10 interpolation endpoint seed{seed}")
    train_analysis = analyze_split(labels_train, scores_train)
    val_analysis = analyze_split(labels_val, scores_val)
    numeric_check(
        val_analysis["grid_scores"][FULL_ID], probes[10], 10)
    e4 = check_e4(args.e4_result, seed, val_analysis["grid_scores"])

    selections = selection_comparison(
        train_analysis, val_analysis, scores_train, scores_val,
        labels_train, labels_val)
    tr_opt = train_analysis["prediction_linear_shrinkage"][
        "analytic_MSE_direction"]
    vl_opt = val_analysis["prediction_linear_shrinkage"][
        "analytic_MSE_direction"]
    print(
        f" seed{seed}: prediction-space λ*_train="
        f"{tr_opt['optimum_clipped_0_to_1']:.4f}; "
        f"descriptive λ*_val={vl_opt['optimum_clipped_0_to_1']:.4f}; "
        f"train-grid selected val ΔMSE-vs-Mean="
        f"{selections['train_MSE_best_two_dim_grid']['validation_delta_MSE_vs_mean']:+.7f}; "
        f"val ΔSpearman-vs-Mean="
        f"{selections['train_MSE_best_two_dim_grid']['validation_delta_Spearman_vs_mean']:+.6f}",
        flush=True)
    result = {
        "seed":seed,
        "reference_mean":str(reference),
        "E0_replay_checks":replay,
        "E4_metric_crosscheck":e4,
        "initial_pool_mean_max_error":identity_error,
        "composition_max_abs_error":max(error_train,error_val),
        "train":train_analysis,
        "validation":val_analysis,
        "train_only_selections_evaluated_on_validation":selections,
    }
    del model, optimizer, online_loader, val_loader, train_eval_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def summarize(rows):
    summary = {"seeds":list(SEEDS)}
    for name in (
        "train_MSE_best_joint_path",
        "train_MSE_best_two_dim_grid",
        "train_Spearman_best_joint_path",
        "train_Spearman_best_two_dim_grid",
        "train_MSE_analytic_prediction_shrinkage",
    ):
        samples = [rows[str(seed)][
            "train_only_selections_evaluated_on_validation"][name]
            for seed in SEEDS]
        summary[name] = {}
        for field in (
            "validation_delta_MSE_vs_mean",
            "validation_delta_MSE_vs_original_full",
            "validation_delta_Spearman_vs_mean",
            "validation_delta_Spearman_vs_original_full",
        ):
            xs = np.asarray([s[field] for s in samples],dtype=np.float64)
            summary[name][field] = {
                "mean":float(xs.mean()),
                "std_across_seeds":float(xs.std(ddof=1)),
                "number_seeds_improved":int(np.sum(xs < 0)) if (
                    "MSE" in field) else int(np.sum(xs > 0)),
                "per_seed":{str(seed):float(x)
                            for seed,x in zip(SEEDS,xs)},
            }
        summary[name]["selected_coefficients"] = {
            str(seed):{k: float(item[k]) for k in ("alpha","beta","lambda")
                       if k in item}
            for seed,item in zip(SEEDS,samples)}
    for split in ("train","validation"):
        analytic = [rows[str(seed)][split][
            "prediction_linear_shrinkage"]["analytic_MSE_direction"]
            for seed in SEEDS]
        summary[f"{split}_prediction_space_derivatives"] = {
            "mean_derivative_at_mean":float(np.mean([
                x["mse_derivative_at_mean_lambda0"] for x in analytic])),
            "mean_derivative_at_full":float(np.mean([
                x["mse_derivative_at_full_lambda1"] for x in analytic])),
            "n_direction_initially_helpful":int(sum(
                x["mse_derivative_at_mean_lambda0"] < 0 for x in analytic)),
            "n_optimum_strictly_inside_0_1":int(sum(
                0 < x["optimum_unconstrained"] < 1 for x in analytic)),
            "per_seed_unconstrained_optimum": {
                str(seed): float(x["optimum_unconstrained"])
                for seed,x in zip(SEEDS,analytic)},
        }
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", default=
                    "/home/data/home/wwr_lumos/models/roberta-base")
    ap.add_argument("--stsb_data_path",default="data/text/stsbenchmark")
    ap.add_argument("--pilot_dir",default=
                    "outputs/text/stsb_frozen_trained_mean")
    ap.add_argument("--e4_result",default=
                    "outputs/text/stsb_mean_equiv_e4_representation_drift/"
                    "e4_five_seed_step10_representation_drift.json")
    ap.add_argument("--output",default=
                    "outputs/text/stsb_mean_equiv_e6_overshoot/"
                    "e6_five_seed_direction_vs_amplitude.json")
    ap.add_argument("--check-only",action="store_true",
                    help="Verify five E0 seed archives and E4 file; no replay.")
    args=ap.parse_args()
    output=Path(args.output)
    if output.exists() and not args.check_only:
        raise FileExistsError(f"E6 result exists, no overwrite: {output}")
    sources=preflight(args)
    e4=Path(args.e4_result)
    if not e4.is_file():
        raise FileNotFoundError(f"Archived E4 comparison missing: {e4}")
    previous=json.loads(e4.read_text(encoding="utf-8"))
    if set(previous["seeds"]) != set(map(str, SEEDS)):
        raise RuntimeError("E4 output incomplete.")
    if args.check_only:
        print("E6 preflight success: all five E0/E4 inputs; writes nothing.",
              flush=True)
        return
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=AutoTokenizer.from_pretrained(
        args.model_path,local_files_only=True)
    print("E6 START: seeds 0..4; only E0 first ten updates/seed; "
          "fixed, predeclared strength grid, no val-based selection.",
          flush=True)
    rows={}
    for seed in SEEDS:
        rows[str(seed)]=run_seed(
            args,seed,tokenizer,sources[seed],device)
    report={
        "experiment":"E6_five_seed_joint_deviation_amplitude_vs_direction",
        "hypothesis":"Is loss of benefit explained by overshooting "
                     "a useful Mean-to-GP direction, or is the direction "
                     "itself harmful/inadequate?",
        "predeclared_strength_grid":list(STRENGTHS),
        "predeclared_joint_path":list(DIAG_STRENGTHS),
        "predeclared_score_shrinkage":list(SCORE_STRENGTHS),
        "method":{
            "gate_strength":
                "m_alpha=m+alpha*(g-m), raw pretrained Mean m "
                "and Step10 FLaG Gate-preprojection g.",
            "projection_strength":
                "z=(1-beta)*m_alpha+beta*(W*m_alpha+b); L2 normalize "
                "then cosine then original Step10 positive-affine Head.",
            "score_shrinkage":
                "s_lambda=s_M+lambda*(s_GP-s_M); lambda chosen analytically "
                "by full train MSE, then evaluated on validation ONCE.",
            "train_selection":
                "argmin TRAIN MSE and argmax TRAIN Spearman from "
                "predeclared grid/path; compare to Mean AND full on val.",
            "batching":
                "Both TRAIN and validation read with eval batch64/dynamic "
                "right padding, unlike E5 TRAIN batch32; original E0 "
                "optimizer replay still uses batch32.",
            "validation_use":
                "Validation metrics are read-only evaluations of "
                "coefficients selected on TRAIN. Validation-derived "
                "oracle optimum is descriptive ONLY, never selected.",
            "original_e0":
                "Exact per-seed trained Mean checkpoint and first ten "
                "original Adam updates; encoder frozen; E0/E4 checks.",
        },
        "seeds":rows,
        "summary":summarize(rows),
        "scientific_decision_rule":[
            "Improvement versus FULL only means partial recovery from "
            "full-model damage; it is NOT a gain over Mean.",
            "A genuine useful deviation requires the TRAIN-selected "
            "strength to improve validation MSE or Spearman vs Mean "
            "across seeds, with appropriate robustness/untouched test "
            "confirmation before claiming generalization.",
            "Prediction-space train optimum strictly inside (0,1) "
            "supports amplitude overshoot on TRAIN in that fixed "
            "score direction; optimum <=0 implies initially harmful "
            "direction, >=1 implies not overshot within the segment.",
            "A function-level grid that beats scalar score shrinkage "
            "on validation at TRAIN-selected strengths suggests "
            "representation geometry may be relevant beyond simple "
            "prediction calibration, but is not a causal proof.",
        ],
        "limitations":[
            "Fixed post-training inference-time strength changes do "
            "not establish why the optimizer took a particular path.",
            "Trained Mean checkpoint selected on same STSB validation "
            "set; independent generalization needs a held-out test.",
            "Five random seeds share the same STSB split; they do not "
            "give five independent datasets.",
            "Train-set-selected parameters are not themselves a "
            "new independently trained model; both evaluation splits "
            "are from the same STSB task.",
            "If no consistent Mean-beating effect arises, this "
            "program SHOULD STOP rather than stack more STSB-only "
            "mechanism tests; review E0-E6 with advisor and pick "
            "new cross-task question/design.",
        ],
    }
    serial=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as f:
        f.write(serial)
    print("\n===== E6 decision summary =====",flush=True)
    for name in (
        "train_MSE_best_joint_path",
        "train_MSE_best_two_dim_grid",
        "train_MSE_analytic_prediction_shrinkage",
    ):
        ss=report["summary"][name]
        m=ss["validation_delta_MSE_vs_mean"]
        s=ss["validation_delta_Spearman_vs_mean"]
        print(f"{name}: val ΔMSE Mean={m['mean']:+.7f}, "
              f"seeds improved {m['number_seeds_improved']}/5; "
              f"val ΔSpearman Mean={s['mean']:+.6f}, "
              f"seeds improved {s['number_seeds_improved']}/5;",
              flush=True)
    for split in ("train","validation"):
        row=report["summary"][f"{split}_prediction_space_derivatives"]
        print(f"{split}: GP-M initially improves MSE in "
              f"{row['n_direction_initially_helpful']}/5 seeds; "
              f"score λ* inside (0,1) in "
              f"{row['n_optimum_strictly_inside_0_1']}/5 seeds",
              flush=True)
    print(f"Saved ONE fixed no-overwrite JSON: {output}",flush=True)


if __name__=="__main__":
    main()
