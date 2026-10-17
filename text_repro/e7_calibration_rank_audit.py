#!/usr/bin/env python3
"""E7 decisive audit: calibration gains vs rank reversals, all five E0 seeds.

Previously completed reports already contain: padding reflection, Gate
saturation/near sentence-independence, DC dominance, Mean+Projection Sprint
baseline, Alignment-aware fusion, and E0-E6 Mean-equivalent drift.
DO NOT repeat any of those ablations.

Here we ask why lower numerical STSB MSE fails to improve Spearman.
Reproduce E0 Step0/1/10 exactly, then for each seed and each predeclared
E6 representation path (Mean, partial(.25,.25), partial(.5,.5), full B2):
  1. Refit ONLY an increasing affine score head using TRAIN MSE (closed form).
     This cannot improve Spearman. Compare calibrated Mean and calibrated
     partial/full B2 on validation. Disentangles ranking from score units.
  2. Count actual pairwise order reversals relative to Mean, distinguishing
     correcting incorrect orders from breaking correct orders, including
     label-gap strata. Kendall-like pairwise audit, NOT Spearman itself.
  3. Exactly decompose raw and calibrated MSE into bias, variance mismatch,
     and Pearson correlation components.
Persist score arrays for both splits for offline follow-up. No additional
model training beyond EXACT E0 ten-update replay; no output overwrites.
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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from e2_adam_step1_counterfactual import compare_with_e0
from e3b_replay_step10_module_splices import numeric_check
from e4_five_seed_representation_drift import SEEDS, preflight
from e6_direction_vs_amplitude import (
    MEAN_ID, FULL_ID, POINTS, point_id, evaluate_path_grid, score_stats, check_e4,
)
from explore_mean_equiv_trajectory import evaluate, get_loaders, get_splits
from train_author_text_protocol import PairCollator, PairDataset, seed_everything
from train_stsb_frozen_mean_start import (
    assert_initial_pool_identity, build_frozen_b2,
)

PARTIAL25=point_id(.25,.25)
PARTIAL50=point_id(.5,.5)
PATHS={
    "trained_mean":MEAN_ID,
    "partial25":PARTIAL25,
    "partial50":PARTIAL50,
    "full_B2":FULL_ID,
}
RANK_VARIANTS=("partial25","partial50","full_B2")
# STSB targets use original 0..5 units; exactly these label separation bins.
LABEL_GAPS=(0., .5, 1., 2., 5.00000001)


def calibrate_increasing_affine(train_raw, label):
    """TRAIN-only least squares score calibration a*s+b with a>0."""
    x=np.asarray(train_raw,dtype=np.float64)
    t=np.asarray(label,dtype=np.float64)/5.
    vx=float(np.mean((x-x.mean())**2))
    if vx<=1e-20:
        raise RuntimeError("Uninformative constant cosine/score predictor.")
    slope=float(np.mean((x-x.mean())*(t-t.mean()))/vx)
    # Same increasing affine hypothesis as original positive exp(log_scale).
    if slope<=0:
        raise RuntimeError(
            "Best affine slope is non-positive; no increasing calibration.")
    intercept=float(t.mean()-slope*x.mean())
    return {"slope":slope,"intercept":intercept,
            "fitted_split":"train","slope_positive":True}


def calibrated_scores(raw, fit):
    return fit["slope"]*np.asarray(raw,dtype=np.float64)+fit["intercept"]


def mse_terms(label, score):
    """Exact decomposition, population SD, and Pearson coefficient."""
    y=np.asarray(label,dtype=np.float64)/5.
    x=np.asarray(score,dtype=np.float64)
    mu=float(x.mean()-y.mean())
    sd_x=float(x.std(ddof=0))
    sd_y=float(y.std(ddof=0))
    corr=float(np.corrcoef(x,y)[0,1])
    bias=mu*mu
    scale=(sd_x-sd_y)**2
    correlation=2*sd_x*sd_y*(1-corr)
    mse=float(np.mean((x-y)**2))
    if abs(bias+scale+correlation-mse)>1e-10:
        raise RuntimeError("MSE bias/scale/Pearson decomposition failed.")
    return {
        "MSE":mse,"squared_mean_bias":bias,
        "variance_scale_mismatch":scale,
        "Pearson_correlation_deficit":correlation,
        "pearson":corr,
        "mean_signed_bias":mu,
        "score_std":sd_x,"target_std":sd_y,
    }


def pairwise_audit(labels, mean_score, candidate_score):
    """Count corrected vs broken orders among pairs of STSB examples.

    A 'ranking comparison' here compares labels of TWO EXAMPLES, each
    example itself being a sentence pair. Spearman is additionally
    reported separately. Tied target labels excluded from flip counts.
    """
    y=np.asarray(labels,dtype=np.float64)
    m=np.asarray(mean_score,dtype=np.float64)
    c=np.asarray(candidate_score,dtype=np.float64)
    n=len(y)
    i,j=np.triu_indices(n,k=1)
    delta_y=y[i]-y[j]
    relevant=(delta_y!=0.)
    # Filter before allocating downstream sign arrays.
    deltas=np.abs(delta_y[relevant])
    desired=np.sign(delta_y[relevant])
    sm=np.sign(m[i[relevant]]-m[j[relevant]])
    sc=np.sign(c[i[relevant]]-c[j[relevant]])
    was_right=sm==desired
    now_right=sc==desired
    corrected=(~was_right)&now_right
    broken=was_right&(~now_right)
    both_wrong=(~was_right)&(~now_right)
    both_right=was_right&now_right
    # Ranking ties should be tracked explicitly, not silently equated
    # with a correct strict ordering.
    tie_m=(sm==0.)
    tie_c=(sc==0.)
    total=int(relevant.sum())
    buckets={}
    for lo,hi in zip(LABEL_GAPS[:-1],LABEL_GAPS[1:]):
        idx=(deltas>lo)&(deltas<=hi)
        cnt=int(idx.sum())
        buckets[f"({lo:g},{min(hi,5.):g}]"]={
            "n_comparisons":cnt,
            "corrected_mean_wrong_to_right":int(corrected[idx].sum()),
            "broken_mean_right_to_wrong":int(broken[idx].sum()),
            "net_corrected_minus_broken":int(
                corrected[idx].sum()-broken[idx].sum()),
        }
    rm=rankdata(m,method="average")
    rc=rankdata(c,method="average")
    return {
        "n_examples":n,"n_comparable_pairs":total,
        "excluded_equal_target_pairs":int((~relevant).sum()),
        "mean_correct_order_fraction":float(was_right.mean()),
        "candidate_correct_order_fraction":float(now_right.mean()),
        "corrected_mean_wrong_to_right":int(corrected.sum()),
        "broken_mean_right_to_wrong":int(broken.sum()),
        "net_corrected_minus_broken":int(corrected.sum()-broken.sum()),
        "remaining_both_wrong":int(both_wrong.sum()),
        "remaining_both_right":int(both_right.sum()),
        "mean_prediction_ties":int(tie_m.sum()),
        "candidate_prediction_ties":int(tie_c.sum()),
        "mean_abs_rank_displacement":float(np.mean(np.abs(rc-rm))),
        "number_examples_rank_changed":int(np.sum(rc!=rm)),
        "label_gap_strata":buckets,
        "note":"Pairwise concordance is Kendall-like, not Spearman; "
               "the latter is reported separately using scipy.",
    }


def split_results(label, scores, fits):
    result={}
    for name, key in PATHS.items():
        s=scores[key]
        calibrated=calibrated_scores(s,fits[name])
        raw=score_stats(label,s)
        fitted=score_stats(label,calibrated)
        # Increasing affine score calibration cannot change ranks.
        if abs(raw["spearman"]-fitted["spearman"])>1e-12:
            raise RuntimeError("Increasing calibration changed Spearman.")
        result[name]={
            "original_head_metrics":raw,
            "refitted_head_metrics":fitted,
            "original_mse_decomposition":mse_terms(label,s),
            "refitted_mse_decomposition":mse_terms(label,calibrated),
            "calibration_MSE_reduction":raw["mse"]-fitted["mse"],
        }
    for name in RANK_VARIANTS:
        audit=pairwise_audit(
            label,scores[MEAN_ID],scores[PATHS[name]])
        result[name]["pairwise_rank_changes_vs_mean"]=audit
        result[name]["relative_to_original_mean"]={
            "original_head_delta_MSE":(
                result[name]["original_head_metrics"]["mse"]-
                result["trained_mean"]["original_head_metrics"]["mse"]),
            "refitted_heads_delta_MSE":(
                result[name]["refitted_head_metrics"]["mse"]-
                result["trained_mean"]["refitted_head_metrics"]["mse"]),
            "delta_Spearman":(
                result[name]["original_head_metrics"]["spearman"]-
                result["trained_mean"]["original_head_metrics"]["spearman"]),
        }
    return result


def validate_against_E6(e6_path, seed, train, val):
    p=json.loads(Path(e6_path).read_text(encoding="utf-8"))
    prior=p["seeds"][str(seed)]
    biggest=0.
    for split, current in (("train",train),("validation",val)):
        for name,key in PATHS.items():
            expected=prior[split]["grid_scores"][key]
            observed=current[name]["original_head_metrics"]
            for metric in ("mse","pearson","spearman"):
                diff=abs(float(expected[metric])-float(observed[metric]))
                biggest=max(biggest,diff)
                if diff>5e-6:
                    raise RuntimeError(
                        f"E7/E6 {seed} {split} {name} {metric} "
                        f"mismatch: {diff}")
    return {"status":"passed","largest_abs_metric_diff":biggest}


def run_seed(args,seed,tokenizer,source,device):
    pilot,_,metadata,probes,reference=source
    args.seed=seed
    train,val=get_splits("stsb",args.stsb_data_path,"",seed,0,0)
    if len(train)!=metadata["n_train"] or len(val)!=metadata["n_validation"]:
        raise RuntimeError(f"Seed {seed} STSB data changed.")
    replay_train,val_loader=get_loaders(
        "stsb",tokenizer,train,val,seed)
    state=torch.load(reference,map_location="cpu",weights_only=True)
    model=build_frozen_b2(args,device,state)
    del state
    pool_initial_diff=assert_initial_pool_identity(
        model,val_loader,device)
    ev0,arr0=evaluate(model,val_loader,"stsb",device)
    checks={"step0":compare_with_e0(
        arr0,pilot/"validation_pairs_step_0.npz",
        f"E7 seed{seed} E0 Step0")}
    numeric_check(ev0["full"],probes[0],0)
    optimizer=torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,weight_decay=0.)
    seed_everything(seed+100000)
    model.train()
    online=[]
    for step,(ta,tb,y) in enumerate(replay_train,1):
        if step>10: break
        ta={k:v.to(device) for k,v in ta.items()}
        tb={k:v.to(device) for k,v in tb.items()}
        optimizer.zero_grad(set_to_none=True)
        pred,_=model(ta,tb)
        loss=F.mse_loss(pred,y.to(device)/5.)
        loss.backward()
        optimizer.step()
        online.append(float(loss.detach().cpu()))
        if step in (1,10):
            ev,arr=evaluate(model,val_loader,"stsb",device)
            checks[f"step{step}"]=compare_with_e0(
                arr,pilot/f"validation_pairs_step_{step}.npz",
                f"E7 seed{seed} E0 Step{step}")
            numeric_check(ev["full"],probes[step],step)
            model.train()
    if len(online)!=10:
        raise RuntimeError(f"Seed {seed}: exact ten replay steps missing.")
    for step,idx in ((1,0),(10,9)):
        if abs(online[idx]-
               float(probes[step]["last_minibatch_train_mse"]))>1e-7:
            raise RuntimeError(f"Seed {seed} E0 online loss mismatch.")

    tr_loader=DataLoader(
        PairDataset(train,"stsb"),batch_size=64,shuffle=False,
        collate_fn=PairCollator(
            tokenizer,max_length=128,label_dtype=torch.float32),
        num_workers=0)
    ytr,str_,_cos_tr,_err_tr=evaluate_path_grid(
        model,tr_loader,device)
    yval,sval,cos_val,_err_val=evaluate_path_grid(
        model,val_loader,device)
    checks["full_from_grid"]=compare_with_e0({
        "labels":yval,
        "score_full":sval[FULL_ID].astype(np.float32),
        "cos_full":cos_val,
    },pilot/"validation_pairs_step_10.npz",
      f"E7 seed{seed} E0 Step10 grid endpoint")
    numeric_check(score_stats(yval,sval[FULL_ID]),probes[10],10)
    e4=check_e4(args.e4_result,seed,
               {key:score_stats(yval,score) for key,score in sval.items()})

    fits={name:calibrate_increasing_affine(
        str_[key],ytr) for name,key in PATHS.items()}
    train_report=split_results(ytr,str_,fits)
    val_report=split_results(yval,sval,fits)
    match_e6=validate_against_E6(
        args.e6_result,seed,train_report,val_report)

    summary={
        "seed":seed,
        "E0_exact_replay":checks,
        "E4_endpoint_crosscheck":e4,
        "E6_train_validation_crosscheck":match_e6,
        "identity_pool_max_error":pool_initial_diff,
        "train_fitted_affine_score_heads":fits,
        "train":train_report,
        "validation":val_report,
    }
    cache={
        "train_labels":ytr,
        "validation_labels":yval,
        "grid_ids":np.array(list(str_.keys())),
        "train_raw_score_matrix":np.stack(
            [str_[key] for key in str_],axis=0),
        "validation_raw_score_matrix":np.stack(
            [sval[key] for key in str_],axis=0),
    }
    for name in RANK_VARIANTS:
        a=val_report[name]["pairwise_rank_changes_vs_mean"]
        print(
            f" [seed {seed}] {name} val: corrected={a['corrected_mean_wrong_to_right']}, "
            f"broken={a['broken_mean_right_to_wrong']}; "
            f"calibrated ΔMSE vs calibrated Mean="
            f"{val_report[name]['relative_to_original_mean']['refitted_heads_delta_MSE']:+.7f}",
            flush=True)
    del model,optimizer,tr_loader,replay_train,val_loader
    gc.collect()
    if device.type=="cuda":torch.cuda.empty_cache()
    return summary,cache


def aggregate(rows):
    result={"seed_ids":list(SEEDS)}
    for split in ("train","validation"):
        result[split]={}
        for name in PATHS:
            sample=[rows[str(seed)][split][name] for seed in SEEDS]
            fields={
                "raw_mse":"original_head_metrics.mse",
                "refitted_mse":"refitted_head_metrics.mse",
                "spearman":"original_head_metrics.spearman",
                "mse_calibration_recovery":"calibration_MSE_reduction",
            }
            if name in RANK_VARIANTS:
                fields.update({
                    "raw_delta_MSE_vs_mean":
                        "relative_to_original_mean.original_head_delta_MSE",
                    "refitted_delta_MSE_vs_refitted_mean":
                        "relative_to_original_mean.refitted_heads_delta_MSE",
                    "delta_Spearman_vs_mean":
                        "relative_to_original_mean.delta_Spearman",
                    "rank_net_corrections":
                        "pairwise_rank_changes_vs_mean.net_corrected_minus_broken",
                })
            d={}
            for field,path in fields.items():
                values=[]
                for row in sample:
                    z=row
                    for part in path.split("."):z=z[part]
                    values.append(float(z))
                ar=np.asarray(values,dtype=np.float64)
                d[field]={
                    "mean":float(ar.mean()),
                    "std":float(ar.std(ddof=1)),
                    "per_seed":dict(zip(
                        map(str,SEEDS),[float(v) for v in ar])),
                }
            result[split][name]=d
    return result


def write_exclusive(output,report,cache):
    out=Path(output)
    arrays=out.with_suffix(".npz")
    if out.exists() or arrays.exists():
        raise FileExistsError(
            f"E7 outputs already exist: {out}, {arrays}. No overwrite.")
    out.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(report,ensure_ascii=False,indent=2,
                    allow_nan=False)+"\n"
    # The ONE cached NPZ enables future analysis without more GPU replay.
    with arrays.open("xb") as f:
        np.savez_compressed(f,**cache)
    with out.open("x",encoding="utf-8") as f:
        f.write(text)
    print(f"Saved E7 report: {out}",flush=True)
    print(f"Saved E7 offline score cache: {arrays}",flush=True)


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
    ap.add_argument("--e6_result",default=
                    "outputs/text/stsb_mean_equiv_e6_overshoot/"
                    "e6_five_seed_direction_vs_amplitude.json")
    ap.add_argument("--output",default=
                    "outputs/text/stsb_mean_equiv_e7_calibration_rank/"
                    "e7_five_seed_calibration_ranking_audit.json")
    ap.add_argument("--check-only",action="store_true")
    args=ap.parse_args()
    output=Path(args.output)
    if (output.exists() or output.with_suffix(".npz").exists()) and not args.check_only:
        raise FileExistsError(f"E7 outputs already exist: {output}")
    sources=preflight(args)
    for p in (args.e4_result,args.e6_result):
        if not Path(p).is_file():
            raise FileNotFoundError(f"Required prior experiment: {p}")
    if args.check_only:
        print("E7 CHECK-ONLY all five E0 inputs and E4/E6 files exist. "
              "No training, no output.",flush=True)
        return
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=AutoTokenizer.from_pretrained(
        args.model_path,local_files_only=True)
    rows={}
    cache={}
    for seed in SEEDS:
        print(f"[E7] Seed={seed}, original E0 10-step replay",flush=True)
        row,arrays=run_seed(args,seed,tok,sources[seed],device)
        rows[str(seed)]=row
        for key,value in arrays.items():
            cache[f"seed_{seed}_{key}"]=value
    report={
        "experiment":"E7_calibration_vs_ranking_diagnostic_five_seeds",
        "scope":"STSB trained-Mean equivalent, frozen RoBERTa, "
                "joint E0 Step10, all seed 0..4; NO new optimization.",
        "research_question":"When MSE decreases but Spearman does not, "
                            "is benefit mostly positive-affine calibration, "
                            "or is there task-relevant ranking information?",
        "definitions":{
            "calibration":"OLS increasing affine map applied to original "
                          "model score, fitted on full TRAIN, evaluated "
                          "unchanged on validation. Must preserve Spearman.",
            "mse_exact":"MSE = (mean prediction - mean label)^2 "
                        "+ (std prediction - std label)^2 "
                        "+ 2*std_prediction*std_label*(1-Pearson).",
            "ordering":"For two STSB sentence-pair examples, compare "
                       "their target relative order to their score "
                       "order. Exclude exactly tied gold labels. "
                       "Count mean-wrong to model-right (corrected) and "
                       "mean-right to model-wrong (broken). "
                       "This is NOT Spearman.",
            "path":"Mean, E6 (.25,.25) and (.5,.5) raw embedding "
                   "interpolations, and full Step10 B2. "
                   "No validation-based selection.",
            "cache":"All raw E6 grid scores for TRAIN and validation "
                    "are saved as a single compressed NPZ, so offline "
                    "rank/calibration reanalysis needs no GPU replay.",
        },
        "seeds":rows,
        "five_seed_summary":aggregate(rows),
        "interpretation_rules":[
            "If recalibrated Mean matches or beats partial-model validation "
            "MSE, the partial-model numerical gain does not establish "
            "valuable new pooling information. This alone does not "
            "mathematically prove all changes are only calibration.",
            "Any increasing affine calibration preserves Spearman EXACTLY; "
            "rank corrections vs rank breaks identify which ordering "
            "comparisons were genuinely changed.",
            "If rank corrections exceed breaks but Spearman worsens, "
            "inspect rank *magnitude* and tied labels: Kendall-style "
            "count and Spearman penalize ordering differences differently.",
            "Even a validation improvement versus mean must be interpreted "
            "cautiously because pretrained Mean was selected on this "
            "very development split.",
            "This is the final STSB-only retrospective diagnostic; "
            "subsequent work should test a substantive, predeclared "
            "ranking-aware model across STSB and Sprint, or discuss "
            "new directions with supervisor rather than stack ablations.",
        ],
        "limitations":[
            "Refitted affine Heads are trained on the same original "
            "TRAIN split from which RoBERTa was fine-tuned, but "
            "never on validation labels.",
            "Five seeds share STSB data and validation-selected Mean; "
            "not five independent datasets or untouched generalization.",
            "The original E0/E6 score Head stays fixed for original rows; "
            "affine recalibration deliberately refits only 2 scalar "
            "coefficients to separate score quality and ordering.",
            "Exact ordering over STSB examples is descriptive and not "
            "a differentiable Spearman training objective.",
        ],
    }
    write_exclusive(output,report,cache)
    for name in PATHS:
        r=report["five_seed_summary"]["validation"][name]
        print(f"{name:<14} raw MSE={r['raw_mse']['mean']:.6f} "
              f"recalibrated MSE={r['refitted_mse']['mean']:.6f} "
              f"Spearman={r['spearman']['mean']:.6f}",flush=True)


if __name__=="__main__":
    main()
