#!/usr/bin/env python3
"""E5: why two Step-10 Mean-deviating blocks fail to yield stable gains.

For seeds 0..4, EXACTLY replay E0's first ten joint Adam updates and examine
TRAIN and validation pairwise predictions under one common Step10 Head:
Mean (M), trained Gate+Latent only (G), trained Projection only (P),
and both (GP). At each pair, decompose the squared-error interaction

I = L(GP)-L(G)-L(P)+L(M)
  = 2*dG*dP + 2*(r+dG+dP)*kappa + kappa**2

r=M-target; dG=G-M; dP=P-M; kappa=GP-G-P+M.

The 2*dG*dP term exists even for *additive* prediction changes. The second
term is the incremental effect of functional nonadditivity at the trained
checkpoint. Neither implies optimization interference or causal task gains.

One command runs all 5 seeds and writes one exclusive JSON. The only
optimization is a ten-minibatch EXACT E0 replay per seed, not new fitting.
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
from e4_five_seed_representation_drift import (
    SEEDS, VARIANTS, preflight,
)
from explore_mean_equiv_trajectory import (
    evaluate, get_loaders, get_splits, metrics,
)
from train_author_text_protocol import (
    PairCollator, PairDataset, seed_everything,
)
from train_stsb_frozen_mean_start import (
    assert_initial_pool_identity, build_frozen_b2,
)
from AMPCliff.factory.pooling.llm_pooling_dropin import masked_mean_pooling


def average(x):
    return float(np.mean(x))


def fraction(x):
    return average(np.asarray(x, dtype=np.float64))


def descriptive(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": average(x),
        "mean_abs": average(np.abs(x)),
        "median": float(np.median(x)),
        "p90_abs": float(np.percentile(np.abs(x), 90)),
        "p99_abs": float(np.percentile(np.abs(x), 99)),
        "max_abs": float(np.max(np.abs(x))),
    }


@torch.no_grad()
def predict_four(model, loader, device):
    """Compute all four outputs from SAME frozen RoBERTa hidden states.

    E0/E3b/E4's exact Gate preprojection output is used. This avoids
    rebuilding Gate parameters or independent models. Identical Step10 Head.
    """
    model.eval()
    parts = {k: [] for k in VARIANTS}
    cosine_parts = {k: [] for k in VARIANTS}
    labels_all = []
    max_functional_composition_error = 0.
    for tok_a, tok_b, y in loader:
        sides = []
        labels_all.append(y.detach().cpu().numpy().astype(np.float32))
        for tokens in (tok_a, tok_b):
            tokens = {k: v.to(device) for k, v in tokens.items()}
            mask = tokens["attention_mask"]
            hidden = model.encoder(**tokens).last_hidden_state
            raw_mean = masked_mean_pooling(hidden, mask, eps=1e-6)
            raw_full = model.pool(hidden, attention_mask=mask)
            raw_gate = model.pool._last_pooled_pre_projection
            raw_projection = model.pool.time_out_proj(raw_mean)
            error = float(torch.max(torch.abs(
                raw_full - model.pool.time_out_proj(raw_gate))).item())
            max_functional_composition_error = max(
                max_functional_composition_error, error)
            if max_functional_composition_error > 2e-6:
                raise RuntimeError(
                    "FLaG_B2 full is not Projection(Gate-output): "
                    "unexpected dropout/norm/forward protocol.")
            sides.append({
                "mean": F.normalize(raw_mean, p=2, dim=-1),
                "gate_latent_only": F.normalize(raw_gate, p=2, dim=-1),
                "projection_only": F.normalize(
                    raw_projection, p=2, dim=-1),
                "full_step10": F.normalize(raw_full, p=2, dim=-1),
            })
        for variant in VARIANTS:
            cos = (sides[0][variant] * sides[1][variant]).sum(dim=-1)
            pred = model.log_scale.exp() * cos + model.bias
            cosine_parts[variant].append(
                cos.detach().cpu().numpy().astype(np.float32))
            parts[variant].append(
                pred.detach().cpu().numpy().astype(np.float32))
    labels = np.concatenate(labels_all)
    preds = {k: np.concatenate(v).astype(np.float64)
             for k, v in parts.items()}
    cosine = {k: np.concatenate(v) for k, v in cosine_parts.items()}
    if any(not np.isfinite(v).all() for v in preds.values()):
        raise RuntimeError("NaN/Inf predictions.")
    if any(v.shape != labels.shape for v in preds.values()):
        raise RuntimeError("Prediction/label ordering mismatch.")
    return labels, preds, cosine, max_functional_composition_error


def loss_decomposition(labels, pred):
    """Exact pairwise squared-error decomposition, then distribution stats."""
    t = labels.astype(np.float64) / 5.0
    m = pred["mean"]
    g = pred["gate_latent_only"]
    p = pred["projection_only"]
    gp = pred["full_step10"]
    r = m - t
    dg, dp = g - m, p - m
    kappa = gp - g - p + m
    radd = m + dg + dp - t

    lm = r * r
    lg = (g-t)**2
    lp = (p-t)**2
    lj = (gp-t)**2
    ladd = radd * radd

    # Two self-interactions are inherent in the squared-error metric;
    # the *prediction* need not contain a nonlinear interaction.
    algebraic = 2. * dg * dp
    nonlinear = 2. * (r + dg + dp) * kappa + kappa * kappa
    total = lj - lg - lp + lm
    if np.max(np.abs(total - algebraic - nonlinear)) > 1e-12:
        raise RuntimeError("Exact per-pair MSE decomposition failed.")
    if np.max(np.abs(lj - lm - ((lg-lm)+(lp-lm)+total))) > 1e-12:
        raise RuntimeError("Exact full-vs-Mean MSE balance failed.")
    if np.max(np.abs(nonlinear - (lj-ladd))) > 1e-12:
        raise RuntimeError("Nonadditivity term must equal L_joint-L_additive.")

    threshold = 1e-10
    sign_eps = 1e-8
    positive_harm = np.maximum(lj-lm, 0.)
    joint_change = gp-m
    # Marginal benefit is relative to SAME Mean, same Step10 Head.
    improvement_g = lg < lm-threshold
    improvement_p = lp < lm-threshold
    improvement_joint = lj < lm-threshold
    categories = {
        "both_individually_better": improvement_g & improvement_p,
        "only_gate_better": improvement_g & ~improvement_p,
        "only_projection_better": ~improvement_g & improvement_p,
        "neither_individually_better": ~improvement_g & ~improvement_p,
    }
    segments = {}
    for label, mask in categories.items():
        segments[label] = {
            "fraction_pairs": fraction(mask),
            "count_pairs": int(mask.sum()),
            "joint_fraction_better_than_mean":
                fraction(improvement_joint[mask]) if mask.any() else None,
            "mean_joint_minus_mean_squared_error":
                average((lj-lm)[mask]) if mask.any() else None,
        }

    # Concentration of positive full-model MSE damage; a few severe
    # failures may outweigh many small pairwise gains.
    harm_total = float(positive_harm.sum())
    n_top = max(1, int(np.ceil(.05 * len(t))))
    top_harm = float(np.sort(positive_harm)[-n_top:].sum())
    report = {
        "n_pairs": int(len(t)),
        "same_head_validation_or_train_metrics": {
            k: metrics("stsb", labels, pred[k]) for k in VARIANTS
        },
        "mse_deltas_vs_same_head_mean": {
            "gate": average(lg-lm),
            "projection": average(lp-lm),
            "joint": average(lj-lm),
            "additive_prediction": average(ladd-lm),
        },
        "mse_interaction_exact": {
            "observed_loss_interaction":
                average(total),
            "algebraic_from_prediction_alignment_2_dg_dp":
                average(algebraic),
            "increment_due_to_nonlinear_prediction_kappa":
                average(nonlinear),
            "closure_abs_error": abs(average(total) -
                                 average(algebraic) -
                                 average(nonlinear)),
            "interpretation": (
                "observed = algebraic 2*dG*dP + nonlinear correction "
                "2*(M-target+dG+dP)*kappa + kappa^2; "
                "the former exists with perfectly additive predictions"),
        },
        "prediction_differences": {
            "d_gate": descriptive(dg),
            "d_projection": descriptive(dp),
            "d_joint": descriptive(joint_change),
            "kappa_joint_minus_additive": descriptive(kappa),
            "mean_pairwise_product_dg_dp": average(dg*dp),
            "pearson_delta_gate_vs_projection": float(
                np.corrcoef(dg, dp)[0, 1]),
            "fraction_gate_projection_same_sign": fraction(
                dg*dp > sign_eps),
            "fraction_gate_projection_opposite_sign": fraction(
                dg*dp < -sign_eps),
            "fraction_gate_moves_toward_label_first_order":
                fraction(r*dg < -sign_eps),
            "fraction_projection_moves_toward_label_first_order":
                fraction(r*dp < -sign_eps),
            "fraction_joint_moves_toward_label_first_order":
                fraction(r*joint_change < -sign_eps),
        },
        "squared_error_outcomes": {
            "fraction_gate_improved": fraction(improvement_g),
            "fraction_projection_improved": fraction(improvement_p),
            "fraction_joint_improved": fraction(improvement_joint),
            "fraction_both_individually_improved_but_joint_worsened":
                fraction(improvement_g & improvement_p & (lj>lm+threshold)),
            "fraction_joint_worse_than_both_components":
                fraction((lj>lg+threshold) & (lj>lp+threshold)),
            "mean_positive_joint_vs_mean_loss_increase":
                average(positive_harm),
            "top_5_percent_pairs_share_of_total_positive_damage":
                top_harm/harm_total if harm_total>0 else None,
            "max_joint_minus_mean_squared_error":
                float(np.max(lj-lm)),
            "mean_joint_minus_mean_squared_error_worst_5_percent":
                float(np.mean(np.sort(lj-lm)[-n_top:])),
            "categories": segments,
        },
        "algebraic_pairwise_closure_max_error":
            float(np.max(np.abs(total-algebraic-nonlinear))),
    }
    return report


def crosscheck_e4(path, seed, val):
    if not path.is_file():
        raise FileNotFoundError(
            "E5 requires existing five-seed E4 JSON for independent "
            "four-model metric cross-check.")
    previous = json.loads(path.read_text(encoding="utf-8"))
    original = previous["seeds"][str(seed)]["counterfactuals"]
    largest = 0.
    for variant in VARIANTS:
        for metric in ("mse", "spearman", "pearson"):
            diff = abs(
                val["same_head_validation_or_train_metrics"][variant][metric]
                - original[variant]["validation"][metric])
            largest = max(largest, diff)
            if diff > 5e-6:
                raise RuntimeError(
                    f"E5/E4 mismatch seed={seed} "
                    f"{variant}:{metric}, delta={diff:g}")
    return {"result": "matched_all_four_E4_variants",
            "max_abs_metric_difference": largest}


def run_one_seed(args, seed, tokenizer, source, device):
    pilot, _summary_path, summary, expected, reference = source
    print(f"\n[E5 seed={seed}] replay Step0,1,10 and check E0...", flush=True)
    args.seed = seed
    train, val = get_splits("stsb", args.stsb_data_path, "", seed, 0, 0)
    if len(train) != summary["n_train"] or len(val) != summary["n_validation"]:
        raise RuntimeError(f"E0 data lengths changed for seed {seed}")
    train_loader, val_loader = get_loaders(
        "stsb", tokenizer, train, val, seed)
    state = torch.load(reference, map_location="cpu", weights_only=True)
    model = build_frozen_b2(args, device, state)
    del state
    identity_err = assert_initial_pool_identity(model, val_loader, device)
    metrics0, arrays0 = evaluate(model, val_loader, "stsb", device)
    replay = {"step0": compare_with_e0(
        arrays0, pilot/"validation_pairs_step_0.npz", "E0 Step0")}
    numeric_check(metrics0["full"], expected[0], 0)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=0.)
    seed_everything(seed+100000)
    model.train()
    online = []
    for step, (a,b,y) in enumerate(train_loader, start=1):
        if step>10:
            break
        a={k:v.to(device) for k,v in a.items()}
        b={k:v.to(device) for k,v in b.items()}
        opt.zero_grad(set_to_none=True)
        pred,_=model(a,b)
        loss=F.mse_loss(pred,y.to(device)/5.)
        loss.backward()
        opt.step()
        online.append(float(loss.detach().cpu()))
        if step in (1,10):
            ev,arr=evaluate(model,val_loader,"stsb",device)
            replay[f"step{step}"] = compare_with_e0(
                arr,pilot/f"validation_pairs_step_{step}.npz",
                f"E0 Step{step} seed{seed}")
            numeric_check(ev["full"],expected[step],step)
            model.train()
    if len(online)!=10:
        raise RuntimeError("Expected exactly ten E0 Adam updates.")
    for step,idx in ((1,0),(10,9)):
        if abs(online[idx] -
               float(expected[step]["last_minibatch_train_mse"]))>1e-7:
            raise RuntimeError(f"E0 step{step} online loss changed: seed{seed}")

    # Deterministic fixed order, not E0's SHUFFLED online minibatches.
    train_eval_loader = DataLoader(
        PairDataset(train,"stsb"),batch_size=32,shuffle=False,
        collate_fn=PairCollator(
            tokenizer,max_length=128,label_dtype=torch.float32),
        num_workers=0)
    yval, pval, cval, error_val = predict_four(
        model,val_loader,device)
    replay["step10_new_prediction_path"] = compare_with_e0(
        {"labels":yval,"score_full":pval["full_step10"].astype(np.float32),
         "cos_full":cval["full_step10"]},
        pilot/"validation_pairs_step_10.npz", f"E0 new Step10 seed{seed}")
    val_result = loss_decomposition(yval,pval)
    numeric_check(
        val_result["same_head_validation_or_train_metrics"]["full_step10"],
        expected[10],10)
    e4_check = crosscheck_e4(
        Path(args.e4_result),seed,val_result)

    print(f"[E5 seed={seed}] evaluating full deterministic TRAIN "
          f"({len(train)} pairs)...",flush=True)
    ytrain, ptrain, _ctrain, error_train = predict_four(
        model,train_eval_loader,device)
    if len(ytrain)!=len(train) or len(yval)!=len(val):
        raise RuntimeError("Train/val pair count mismatch.")
    train_result=loss_decomposition(ytrain,ptrain)

    i_val=val_result["mse_interaction_exact"]
    i_train=train_result["mse_interaction_exact"]
    print(f"  seed={seed}: TRAIN interaction={i_train['observed_loss_interaction']:+.7f} "
          f"(alignment={i_train['algebraic_from_prediction_alignment_2_dg_dp']:+.7f}, "
          f"kappa={i_train['increment_due_to_nonlinear_prediction_kappa']:+.7f}); "
          f"VALIDATION interaction={i_val['observed_loss_interaction']:+.7f} "
          f"(alignment={i_val['algebraic_from_prediction_alignment_2_dg_dp']:+.7f}, "
          f"kappa={i_val['increment_due_to_nonlinear_prediction_kappa']:+.7f})",
          flush=True)
    data={
        "seed": seed,
        "reference_mean_checkpoint": str(reference),
        "identity_max_abs_error": identity_err,
        "exact_E0_replay": replay,
        "existing_E4_crosscheck": e4_check,
        "full_projection_composition_max_abs_error": max(
            error_train,error_val),
        "train":train_result,
        "validation":val_result,
    }
    del model,opt,train_loader,val_loader,train_eval_loader
    gc.collect()
    if device.type=="cuda":
        torch.cuda.empty_cache()
    return data


def aggregate(seeds):
    summary={"seed_ids":list(SEEDS)}
    paths={
        "observed_interaction":
            ("mse_interaction_exact","observed_loss_interaction"),
        "alignment_squared_loss":
            ("mse_interaction_exact",
             "algebraic_from_prediction_alignment_2_dg_dp"),
        "functional_nonadditivity":
            ("mse_interaction_exact",
             "increment_due_to_nonlinear_prediction_kappa"),
        "joint_delta_MSE":
            ("mse_deltas_vs_same_head_mean","joint"),
        "gate_delta_MSE":
            ("mse_deltas_vs_same_head_mean","gate"),
        "projection_delta_MSE":
            ("mse_deltas_vs_same_head_mean","projection"),
        "additive_delta_MSE":
            ("mse_deltas_vs_same_head_mean","additive_prediction"),
        "delta_gate_projection_pearson":
            ("prediction_differences",
             "pearson_delta_gate_vs_projection"),
        "fraction_gate_projection_same_sign":
            ("prediction_differences",
             "fraction_gate_projection_same_sign"),
        "mean_abs_kappa":
            ("prediction_differences","kappa_joint_minus_additive",
             "mean_abs"),
        "fraction_joint_MSE_improved":
            ("squared_error_outcomes","fraction_joint_improved"),
        "fraction_both_improve_joint_worse":
            ("squared_error_outcomes",
             "fraction_both_individually_improved_but_joint_worsened"),
        "top5_harm_concentration":
            ("squared_error_outcomes",
             "top_5_percent_pairs_share_of_total_positive_damage"),
    }
    for split in ("train","validation"):
        summary[split]={}
        for key,path in paths.items():
            def pick(row):
                node=row[split]
                for p in path:
                    node=node[p]
                return node
            values=[pick(seeds[str(seed)]) for seed in SEEDS]
            if any(x is None for x in values):
                summary[split][key]={"mean":None,
                                      "per_seed":dict(zip(
                                          map(str,SEEDS),values))}
                continue
            a=np.asarray(values,dtype=np.float64)
            summary[split][key]={
                "mean":average(a),
                "std_across_seeds":float(a.std(ddof=1)),
                "per_seed":{str(seed):float(value)
                            for seed,value in zip(SEEDS,values)}}
        summary[split]["num_positive_interactions"]=sum(
            seeds[str(seed)][split]["mse_interaction_exact"][
                "observed_loss_interaction"]>0 for seed in SEEDS)
    summary["validation_minus_train_interaction_per_seed"]={
        str(seed):seeds[str(seed)]["validation"][
            "mse_interaction_exact"]["observed_loss_interaction"] -
        seeds[str(seed)]["train"][
            "mse_interaction_exact"]["observed_loss_interaction"]
        for seed in SEEDS
    }
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path",default=
                    "/home/data/home/wwr_lumos/models/roberta-base")
    ap.add_argument("--stsb_data_path",default="data/text/stsbenchmark")
    ap.add_argument("--pilot_dir",default=
                    "outputs/text/stsb_frozen_trained_mean")
    ap.add_argument("--e4_result",default=
                    "outputs/text/stsb_mean_equiv_e4_representation_drift/"
                    "e4_five_seed_step10_representation_drift.json")
    ap.add_argument("--output",default=
                    "outputs/text/stsb_mean_equiv_e5_joint_gain/"
                    "e5_five_seed_step10_joint_gain_decomposition.json")
    ap.add_argument("--check-only",action="store_true",
                    help="Validate all five input seed sources; no training.")
    args=ap.parse_args()
    output=Path(args.output)
    if output.exists() and not args.check_only:
        raise FileExistsError(
            f"E5 result exists, refusing overwrite: {output}")
    sources=preflight(args)
    e4_path=Path(args.e4_result)
    if not e4_path.is_file():
        raise FileNotFoundError(
            f"Need E4 five-seed JSON for validation: {e4_path}")
    prior=json.loads(e4_path.read_text(encoding="utf-8"))
    if set(prior["seeds"]) != set(map(str,SEEDS)):
        raise RuntimeError("E4 JSON does not cover exactly seeds 0-4.")
    print("E5 preflight passed: E0 archives 0-4 and E4 five-seed JSON.",
          flush=True)
    if args.check_only:
        print("CHECK-ONLY: no replay or output.",flush=True)
        return
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=AutoTokenizer.from_pretrained(
        args.model_path,local_files_only=True)
    print("Exact 10 Adam updates per seed; full TRAIN + VALIDATION; "
          "common Step10 Head, no new model fit.",flush=True)
    seeds={}
    for seed in SEEDS:
        seeds[str(seed)]=run_one_seed(
            args,seed,tokenizer,sources[seed],device)
    report={
        "experiment":"E5_five_seed_train_validation_joint_gain_mechanism",
        "seeds_requested":list(SEEDS),
        "phase":"Step10; trained Mean start; frozen RoBERTa",
        "objective":"Explain the positive MSE interaction from E4 by "
                    "separating arithmetic squared-loss alignment from "
                    "prediction-level functional nonadditivity, and "
                    "comparing full train and validation splits.",
        "methods":{
            "prediction_definitions":
                "M = trained RoBERTa masked Mean, G = same jointly trained "
                "Step10 Gate preprojection, P = same jointly trained "
                "Step10 Projection on Mean, GP = full jointly trained B2. "
                "All scores use exactly the same Step10 affine Head.",
            "loss_decomposition":
                "r=M-y; dG=G-M; dP=P-M; k=GP-G-P+M; "
                "L_GP-L_G-L_P+L_M = 2*dG*dP + "
                "2*(r+dG+dP)*k + k^2",
            "train_evaluation":
                "Full, ordered non-shuffled STSB TRAIN set, batch32, "
                "dynamic right padding; NOT online minibatch loss.",
            "validation_evaluation":
                "Official STSB development set, same as E0/E4, batch64.",
            "training":
                "EXACT original E0 Adam seed-matched first 10 batches; "
                "NO independently re-trained Gate/Projection models.",
        },
        "seeds":seeds,
        "five_seed_summary":aggregate(seeds),
        "limitations":[
            "A positive squared-loss interaction can occur without "
            "functional coupling (2*dG*dP alone).",
            "A positive nonlinear-correction contribution does NOT "
            "prove optimizer interference or parameter incompatibility.",
            "Comparisons use the jointly learned Step10 Head and "
            "counterfactual inference paths; no separate optimal Heads.",
            "MSE gains/losses are from 10 optimizer steps, not full "
            "540-step adaptation, and do not establish task-level "
            "conditions for useful deviation across AMP/Sprint.",
            "Mean reference was previously selected using this same "
            "validation split. It is NOT an untouched test set.",
            "A per-pair sign or loss-change concentration is descriptive "
            "of this split and sample distribution, not a causal "
            "explanation of semantic usefulness."
        ],
    }
    content=json.dumps(report,indent=2,ensure_ascii=False,
                       allow_nan=False)+"\n"
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as f:
        f.write(content)
    print("\n===== E5 five-seed decomposition summary =====",flush=True)
    for split in ("train","validation"):
        s=report["five_seed_summary"][split]
        print(f"[{split}] observed I="
              f"{s['observed_interaction']['mean']:+.7f} "
              f"= alignment {s['alignment_squared_loss']['mean']:+.7f} "
              f"+ functional {s['functional_nonadditivity']['mean']:+.7f}; "
              f"joint ΔMSE={s['joint_delta_MSE']['mean']:+.7f}",
              flush=True)
    print(f"Result saved in ONE fixed no-overwrite file: {output}",flush=True)


if __name__=="__main__":
    main()
