#!/usr/bin/env python3
"""E8: algebra-checked structural counterfactuals for E0 Step10 B2.

QUESTION: which operation newly added beyond Mean breaks STSB order?
Exact current B2 identity (masked-mean, no postnorm, no freq-specific gate):
  z = W*[a(X)*m + .5*(a(X)-b(X))*(r-m)] + c
where m = masked mean, r = masked mean of circularly reversed masked
hidden states, a/b = 2*sigmoid Gate real/imag channel multipliers.

Interventions: Gate=identity / TRAIN-frequency-gate average / per-example;
reflection off/on; W=identity/trained; c=zero/trained. This yields 20
unique counterfactuals. All branches use the SAME trained Step10 Head.
Estimate the static Gate on TRAIN only. All five E0 seeds in ONE run.
No model retraining, only exact original ten-batch Adam replay per seed.

Rank damage is attributed by both leave-one-term-out and a 4-factor
Shapley allocation of *measured* Spearman changes, with exact closure.
Neither attribution implies that individual parameters were trained
independently or that the mechanism generalizes to all tasks.

Prerequisites: E0 seed archives and E7 per-pair score cache.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from e2_adam_step1_counterfactual import compare_with_e0
from e3b_replay_step10_module_splices import numeric_check
from e4_five_seed_representation_drift import SEEDS, preflight
from e6_direction_vs_amplitude import score_stats, MEAN_ID, FULL_ID
from e7_calibration_rank_audit import pairwise_audit
from explore_mean_equiv_trajectory import evaluate, get_loaders, get_splits
from train_author_text_protocol import PairCollator, PairDataset, seed_everything
from train_stsb_frozen_mean_start import (
    assert_initial_pool_identity, build_frozen_b2,
)
from AMPCliff.factory.pooling.llm_pooling_dropin import masked_mean_pooling


FACTORS=("gate","reflection","weight","bias")
MODES=("identity","static","dynamic")


def name(mode, r, w, c):
    # No real/imag asymmetry exists under identity a=b=1.
    if mode=="identity":
        r=False
    return f"{mode}_r{int(r)}_w{int(w)}_c{int(c)}"


MEAN=name("identity",False,False,False)
FULL=name("dynamic",True,True,True)
STATIC_FULL=name("static",True,True,True)

# Four independent binary interventions for either fixed trained static
# gate or dynamically recomputed example Gate. Off Gate = identity a=b=1.
CONFIGS=tuple(
    (mode,r,w,c)
    for mode in MODES
    for r in ((False,) if mode=="identity" else (False,True))
    for w in (False,True)
    for c in (False,True)
)
KEYS=tuple(name(*spec) for spec in CONFIGS)
assert len(CONFIGS)==20 and len(set(KEYS))==20


def check_model_protocol(model):
    pool=model.pool
    if (pool.time_pool!="mean" or pool.post_pool_norm or
        pool.gate_parameterization!="centered_sigmoid" or
        getattr(pool,"window_type",None) is not None or
        getattr(pool,"fixed_fft_length",None) is not None or
        getattr(pool,"remove_dc",False) or getattr(pool,"dc_only",False) or
        getattr(pool,"attention_frequency_gate",False) or
        getattr(pool,"learned_frequency_gate",False) or
        not pool.use_gate or not pool.use_latent):
        raise RuntimeError(
            "E8 exact formula only applies to E0 B2: masked-mean, "
            "centered sigmoid Gate, no window/freq-gate/DC override, "
            "no post-pool normalization.")
    if pool.dropout.p!=0.0:
        raise RuntimeError("E8 expected zero dropout.")


def extract_split(model,loader,device):
    """One frozen encoder pass per sentence occurrence, with exact checks.

    All inputs are cached on CPU once: avoids repeatedly running RoBERTa
    to evaluate the structural counterfactuals.
    """
    model.eval()
    side_parts={s:{k:[] for k in ("m","r","a","b")}
                for s in ("left","right")}
    labels=[]
    native_full_score=[]
    native_mean_score=[]
    largest_pre_error=0.
    largest_full_error=0.
    max_T=0
    with torch.no_grad():
        for ta,tb,y in loader:
            labels.append(y.numpy().astype(np.float32))
            per_side=[]
            for side,toks in (("left",ta),("right",tb)):
                toks={k:v.to(device) for k,v in toks.items()}
                mask=toks["attention_mask"]
                hidden=model.encoder(**toks).last_hidden_state
                max_T=max(max_T,int(hidden.size(1)))
                raw_mean=masked_mean_pooling(hidden,mask,eps=1e-6)
                actual_full=model.pool(hidden,attention_mask=mask)
                actual_pre=model.pool._last_pooled_pre_projection
                gate=model.pool._last_raw_gate
                if gate.shape != (hidden.size(0),2*hidden.size(-1)):
                    raise RuntimeError("Unexpected Gate shape.")
                a,b=(2.*gate).chunk(2,dim=-1)
                # The Fourier input has ALREADY been masked to zero.
                x=hidden*mask.unsqueeze(-1).to(hidden.dtype)
                rev_idx=(-torch.arange(x.size(1),device=device))%x.size(1)
                # Apply the ORIGINAL attention mask at output positions
                # after reversing the (masked) FFT input.
                reversed_x=x.index_select(1,rev_idx)
                reversed_mean=masked_mean_pooling(
                    reversed_x,mask,eps=1e-6)
                pre_formula=a*raw_mean + .5*(a-b)*(reversed_mean-raw_mean)
                pre_err=float((actual_pre-pre_formula).abs().max().item())
                largest_pre_error=max(largest_pre_error,pre_err)
                full_formula=F.linear(
                    pre_formula,
                    model.pool.time_out_proj.weight,
                    model.pool.time_out_proj.bias)
                full_err=float((actual_full-full_formula).abs().max().item())
                largest_full_error=max(largest_full_error,full_err)
                if pre_err>2e-5 or full_err>2e-5:
                    raise RuntimeError(
                        f"FFT exact reconstruction FAILED: "
                        f"pre={pre_err:.6g}, full={full_err:.6g}. "
                        f"Stop: do not interpret attribution.")
                for k,tensor in (
                    ("m",raw_mean),("r",reversed_mean),
                    ("a",a),("b",b)):
                    side_parts[side][k].append(
                        tensor.detach().cpu().numpy().astype(np.float32))
                per_side.append((F.normalize(actual_full,p=2,dim=-1),
                                 F.normalize(raw_mean,p=2,dim=-1)))
            full_cos=(per_side[0][0]*per_side[1][0]).sum(-1)
            mean_cos=(per_side[0][1]*per_side[1][1]).sum(-1)
            native_full_score.append((
                model.log_scale.exp()*full_cos+model.bias
            ).cpu().numpy().astype(np.float32))
            native_mean_score.append((
                model.log_scale.exp()*mean_cos+model.bias
            ).cpu().numpy().astype(np.float32))
    data={
        side:{k:np.concatenate(chunks,axis=0)
              for k,chunks in channels.items()}
        for side,channels in side_parts.items()
    }
    y=np.concatenate(labels)
    native={
        "full":np.concatenate(native_full_score),
        "mean":np.concatenate(native_mean_score)
    }
    n=len(y)
    if any(v.shape[0]!=n for d in data.values() for v in d.values()):
        raise RuntimeError("Sentence cache does not align with pairs.")
    return y,data,native,{
        "n_pairs":int(n),"max_fft_length_seen":max_T,
        "n_sentence_occurrences":int(2*n),
        "pre_projection_formula_max_abs_diff":largest_pre_error,
        "full_projection_formula_max_abs_diff":largest_full_error,
    }


def make_mean_gate(train):
    a=np.concatenate([train[s]["a"] for s in ("left","right")])
    b=np.concatenate([train[s]["b"] for s in ("left","right")])
    return a.mean(0,dtype=np.float64).astype(np.float32), (
        b.mean(0,dtype=np.float64).astype(np.float32))


def gate_statistics(split, abar,bbar):
    a=np.concatenate([split[s]["a"] for s in ("left","right")])
    b=np.concatenate([split[s]["b"] for s in ("left","right")])
    std_a=a.std(0,dtype=np.float64)
    std_b=b.std(0,dtype=np.float64)
    return {
        "n_sentence_occurrences":int(a.shape[0]),
        "mean_absolute_real_gate_minus_one":
            float(np.mean(np.abs(a-1.))),
        "mean_absolute_real_imag_difference":
            float(np.mean(np.abs(a-b))),
        "real_gate_mean_across_channel_std":
            float(np.mean(std_a)),
        "imag_gate_mean_across_channel_std":
            float(np.mean(std_b)),
        "real_gate_p90_channel_std":float(np.percentile(std_a,90)),
        "imag_gate_p90_channel_std":float(np.percentile(std_b,90)),
        "real_gate_mean_abs_dynamic_minus_train_static":
            float(np.mean(np.abs(a-abar))),
        "imag_gate_mean_abs_dynamic_minus_train_static":
            float(np.mean(np.abs(b-bbar))),
        "train_static_real_gate_mean":float(np.mean(abar)),
        "train_static_imag_gate_mean":float(np.mean(bbar)),
    }


@torch.no_grad()
def counterfactual_scores(model,cache,abar,bbar,device,batch=96):
    """Evaluate 20 literal static/dynamic/counterfactual variants.

    Retain exactly original trained Projection matrix/offset and Step10
    Head in all variants. Only the indicated sub-operation is bypassed.
    """
    weight=model.pool.time_out_proj.weight.detach()
    bias=model.pool.time_out_proj.bias.detach()
    hscale=model.log_scale.exp().detach()
    hoffset=model.bias.detach()
    n=cache["left"]["m"].shape[0]
    pieces={k:[] for k in KEYS}
    abar_tensor=torch.from_numpy(abar).to(device).unsqueeze(0)
    bbar_tensor=torch.from_numpy(bbar).to(device).unsqueeze(0)
    for lo in range(0,n,batch):
        hi=min(n,lo+batch)
        vectors={}
        for side in ("left","right"):
            inp={k:torch.from_numpy(x[lo:hi]).to(device)
                 for k,x in cache[side].items()}
            m=inp["m"]
            out={}
            for mode,r,w,c in CONFIGS:
                if mode=="identity":
                    a=torch.ones_like(inp["a"])
                    b=torch.ones_like(inp["b"])
                elif mode=="static":
                    a=abar_tensor
                    b=bbar_tensor
                else:
                    a=inp["a"]
                    b=inp["b"]
                z=a*m
                if r:
                    z=z+.5*(a-b)*(inp["r"]-m)
                if w:
                    z=F.linear(z,weight,None)
                if c:
                    z=z+bias
                out[name(mode,r,w,c)]=F.normalize(
                    z,p=2,dim=-1)
            vectors[side]=out
        for key in KEYS:
            cosine=(vectors["left"][key]*vectors["right"][key]).sum(-1)
            score=hscale*cosine+hoffset
            pieces[key].append(score.detach().cpu().numpy().astype(np.float32))
    return {key:np.concatenate(pieces[key]).astype(np.float64)
            for key in KEYS}


def check_against_e7(npz,seed,split,labels,scores,native):
    suffix=f"seed_{seed}_{split}"
    original=npz[f"{suffix}_raw_score_matrix"]
    keys=list(npz[f"seed_{seed}_grid_ids"].astype(str))
    reference=npz[f"{suffix}_labels"]
    if not np.array_equal(labels,reference):
        raise RuntimeError(f"E7 {seed}/{split}: labels/order mismatch.")
    def ref(key):
        return original[keys.index(key)]
    baseline_expected=ref(MEAN_ID)
    full_expected=ref(FULL_ID)
    values={
        "formula_mean_vs_E7":np.max(np.abs(scores[MEAN]-baseline_expected)),
        "formula_full_vs_E7":np.max(np.abs(scores[FULL]-full_expected)),
        "native_mean_vs_E7":np.max(np.abs(native["mean"]-baseline_expected)),
        "native_full_vs_E7":np.max(np.abs(native["full"]-full_expected)),
        "formula_full_vs_native":np.max(np.abs(scores[FULL]-native["full"])),
    }
    if max(values.values())>2e-5:
        raise RuntimeError(
            f"E8/E7 predictions differ {seed}/{split}: {values}")
    return {k:float(v) for k,v in values.items()}


def fc_key(mode, included):
    gate="gate" in included
    return name(mode if gate else "identity",
                "reflection" in included,
                "weight" in included,
                "bias" in included)


def shapley_by_mode(rows,mode,metric="spearman"):
    # Four independently toggled operations, evaluated at a single
    # jointly trained checkpoint. Gate off implies real=imag=1, so
    # reflection has no effect. This interaction is accounted for
    # explicitly in the Shapley allocation.
    final=rows[fc_key(mode,FACTORS)][metric]
    base=rows[MEAN][metric]
    contributions={}
    for factor in FACTORS:
        others=[f for f in FACTORS if f!=factor]
        value=0.
        for n in range(len(others)+1):
            for xs in itertools.combinations(others,n):
                subset=set(xs)
                lower=rows[fc_key(mode,subset)][metric]
                upper=rows[fc_key(mode,subset|{factor})][metric]
                weight=(math.factorial(n) *
                        math.factorial(3-n) / math.factorial(4))
                value+=weight*(upper-lower)
        contributions[factor]=value
    if abs(sum(contributions.values())-(final-base))>1e-10:
        raise RuntimeError("Spearman Shapley additive closure failed.")
    return {
        "baseline_mean":float(base),
        "full_four_factor":float(final),
        "total_delta_Spearman":float(final-base),
        "Shapley_contribution":{k:float(v) for k,v in contributions.items()},
        "closure_abs_error":float(abs(
            sum(contributions.values())-(final-base))),
        "interpretation":"Symmetric allocation of interacting post-hoc "
            "forward operations; not independently retrained modules "
            "or a unique physical causal effect.",
    }


INTERESTING={
    "mean":MEAN,
    "projection_only":name("identity",False,True,True),
    "static_gate_only":name("static",False,False,False),
    "dynamic_gate_only":name("dynamic",False,False,False),
    "static_no_reflection_with_projection":
        name("static",False,True,True),
    "dynamic_no_reflection_with_projection":
        name("dynamic",False,True,True),
    "static_full":STATIC_FULL,
    "full_B2":FULL,
    "full_without_bias":name("dynamic",True,True,False),
    "full_without_projection_weight":
        name("dynamic",True,False,True),
}
RESTORATIONS={
    "remove_entire_gate":
        name("identity",False,True,True),
    "replace_dynamic_gate_with_train_static":STATIC_FULL,
    "remove_reflection":name("dynamic",False,True,True),
    "remove_projection_weight":name("dynamic",True,False,True),
    "remove_projection_bias":name("dynamic",True,True,False),
}


def analyze_split(labels,scores,rank_audit=False):
    rows={key:score_stats(labels,scores[key]) for key in KEYS}
    mean=rows[MEAN]
    full=rows[FULL]
    diag={
        "metrics":rows,
        "mean_vs_full":{
            k:float(full[k]-mean[k])
            for k in ("spearman","pearson","mse")},
        "train_static_vs_dynamic_full":{
            k:float(rows[STATIC_FULL][k]-full[k])
            for k in ("spearman","pearson","mse")},
        "Shapley_dynamic":shapley_by_mode(rows,"dynamic"),
        "Shapley_static":shapley_by_mode(rows,"static"),
        "full_component_removal_effect":{
            key:{
                k:float(rows[variant][k]-full[k])
                for k in ("spearman","pearson","mse")}
            for key,variant in RESTORATIONS.items()},
    }
    if rank_audit:
        # Unlike Spearman, pairwise comparisons exclude tied gold labels.
        diag["rank_vs_mean"]={
            k:pairwise_audit(
                labels,scores[MEAN],scores[variant])
            for k,variant in INTERESTING.items()
            if k!="mean"}
        diag["rank_restored_from_full"]={
            k:pairwise_audit(
                labels,scores[FULL],scores[variant])
            for k,variant in RESTORATIONS.items()}
    return diag


def replay_seed(args,seed,tokenizer,source,device,npz):
    pilot,_,summary,probes,reference=source
    args.seed=seed
    train,val=get_splits(
        "stsb",args.stsb_data_path,"",seed,0,0)
    if len(train)!=summary["n_train"] or len(val)!=summary["n_validation"]:
        raise RuntimeError("E0 split length changed.")
    online_loader,val_loader=get_loaders(
        "stsb",tokenizer,train,val,seed)
    state=torch.load(reference,map_location="cpu",weights_only=True)
    model=build_frozen_b2(args,device,state)
    del state
    check_model_protocol(model)
    initial_err=assert_initial_pool_identity(
        model,val_loader,device)
    eval0,arr0=evaluate(model,val_loader,"stsb",device)
    checks={"step0":compare_with_e0(
        arr0,pilot/"validation_pairs_step_0.npz",
        f"E8 E0 Step0 seed{seed}")}
    numeric_check(eval0["full"],probes[0],0)
    opt=torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,weight_decay=0.)
    seed_everything(seed+100000)
    model.train()
    losses=[]
    for step,(ta,tb,y) in enumerate(online_loader,1):
        if step>10:break
        ta={k:v.to(device) for k,v in ta.items()}
        tb={k:v.to(device) for k,v in tb.items()}
        opt.zero_grad(set_to_none=True)
        pred,_=model(ta,tb)
        loss=F.mse_loss(pred,y.to(device)/5.)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if step in (1,10):
            ev,ar=evaluate(model,val_loader,"stsb",device)
            checks[f"step{step}"]=compare_with_e0(
                ar,pilot/f"validation_pairs_step_{step}.npz",
                f"E8 E0 Step{step} seed{seed}")
            numeric_check(ev["full"],probes[step],step)
            model.train()
    if len(losses)!=10:
        raise RuntimeError("Exact E0 replay did not reach ten steps.")
    for step,ix in ((1,0),(10,9)):
        if abs(losses[ix]-float(
            probes[step]["last_minibatch_train_mse"]))>1e-7:
            raise RuntimeError("Online E0 batch MSE mismatch.")

    # Stable non-shuffle order, same batch size as E7 train evaluation.
    train_eval_loader=DataLoader(
        PairDataset(train,"stsb"),batch_size=64,shuffle=False,
        collate_fn=PairCollator(
            tokenizer,max_length=128,label_dtype=torch.float32),
        num_workers=0)
    print(f"[E8 seed {seed}] caching TRAIN Gate/Mean/reflection...",flush=True)
    ytr,cache_tr,native_tr,valid_tr=extract_split(
        model,train_eval_loader,device)
    a_bar,b_bar=make_mean_gate(cache_tr)
    train_gate_stats=gate_statistics(cache_tr,a_bar,b_bar)
    print(f"[E8 seed {seed}] caching VALIDATION Gate/Mean/reflection...",
          flush=True)
    yval,cache_va,native_va,valid_va=extract_split(
        model,val_loader,device)
    val_gate_stats=gate_statistics(cache_va,a_bar,b_bar)
    if len(ytr)!=5749 or len(yval)!=1500:
        raise RuntimeError("Expected full original STSB train+validation.")

    results={}
    score_archive={}
    for split,y,cache,native,diag,stats in (
        ("train",ytr,cache_tr,native_tr,valid_tr,train_gate_stats),
        ("validation",yval,cache_va,native_va,valid_va,val_gate_stats)):
        print(f"[E8 seed {seed}] {split}: computing structural controls...",
              flush=True)
        scores=counterfactual_scores(
            model,cache,a_bar,b_bar,device)
        checks[f"{split}_vs_E7"]=check_against_e7(
            npz,seed,split,y,scores,native)
        result=analyze_split(
            y,scores,rank_audit=(split=="validation"))
        result["algebraic_checks"]=diag
        result["gate_variation"]=stats
        results[split]=result
        score_archive[f"{split}_labels"]=y
        score_archive[f"{split}_scores"]=np.stack(
            [scores[key] for key in KEYS],axis=0).astype(np.float32)
    score_archive["variant_names"]=np.asarray(KEYS)
    sp=results["validation"]["Shapley_dynamic"]
    print(
        f"  seed{seed} full ΔSpearman={sp['total_delta_Spearman']:+.6f}; "
        f"Shapley Gate={sp['Shapley_contribution']['gate']:+.6f}, "
        f"reflection={sp['Shapley_contribution']['reflection']:+.6f}, "
        f"W={sp['Shapley_contribution']['weight']:+.6f}, "
        f"bias={sp['Shapley_contribution']['bias']:+.6f}; "
        f"static-versus-dynamic="
        f"{results['validation']['train_static_vs_dynamic_full']['spearman']:+.6f}",
        flush=True)
    output={
        "seed":seed,
        "exact_replay_and_archival_checks":checks,
        "initial_pool_identity_max_abs_error":initial_err,
        "train_static_gate_observations":2*len(ytr),
        "train":results["train"],
        "validation":results["validation"],
    }
    del model,opt,online_loader,val_loader,train_eval_loader
    del cache_tr,cache_va
    gc.collect()
    if device.type=="cuda":torch.cuda.empty_cache()
    return output,score_archive


def summary(rows):
    report={"seed_ids":list(SEEDS)}
    for split in ("train","validation"):
        report[split]={}
        paths={
            "total_full_minus_mean_spearman":
                ("mean_vs_full","spearman"),
            "static_minus_dynamic_spearman":
                ("train_static_vs_dynamic_full","spearman"),
            "full_no_reflection_restoration":
                ("full_component_removal_effect",
                 "remove_reflection","spearman"),
            "full_no_projection_weight_restoration":
                ("full_component_removal_effect",
                 "remove_projection_weight","spearman"),
            "full_no_bias_restoration":
                ("full_component_removal_effect",
                 "remove_projection_bias","spearman"),
            "full_static_gate_restoration":
                ("full_component_removal_effect",
                 "replace_dynamic_gate_with_train_static","spearman"),
            "real_gate_mean_across_channel_std":
                ("gate_variation","real_gate_mean_across_channel_std"),
        }
        for factor in FACTORS:
            paths[f"dynamic_shapley_{factor}"]=(
                "Shapley_dynamic","Shapley_contribution",factor)
            paths[f"static_shapley_{factor}"]=(
                "Shapley_static","Shapley_contribution",factor)
        if split=="validation":
            for control in RESTORATIONS:
                paths[f"{control}_net_rank_restored_from_full"]=(
                    "rank_restored_from_full",control,
                    "net_corrected_minus_broken")
        for measure,path in paths.items():
            vals=[]
            for seed in SEEDS:
                v=rows[str(seed)][split]
                for field in path:v=v[field]
                vals.append(float(v))
            arr=np.asarray(vals,dtype=np.float64)
            report[split][measure]={
                "mean":float(arr.mean()),
                "std_over_seeds":float(arr.std(ddof=1)),
                "per_seed":{str(seed):float(value)
                            for seed,value in zip(SEEDS,arr.tolist())}}
    return report


def self_test():
    torch.manual_seed(83)
    for T in (5,6,7,12):
        x=torch.randn(3,T,9,dtype=torch.float64)
        mask=(torch.arange(T)<T-2)[None,:,None]
        x=x*mask
        a=2.*torch.rand(3,9,dtype=torch.float64)
        b=2.*torch.rand(3,9,dtype=torch.float64)
        fft=torch.fft.rfft(x,dim=1)
        reconstructed=torch.fft.irfft(torch.complex(
            fft.real*a[:,None,:],fft.imag*b[:,None,:]),
            n=T,dim=1)
        m=(x*mask).sum(1)/mask.sum(1)
        r=(x.index_select(1,(-torch.arange(T))%T)*mask).sum(1)/mask.sum(1)
        z=a*m+.5*(a-b)*(r-m)
        actual=(reconstructed*mask).sum(1)/mask.sum(1)
        if not torch.allclose(z,actual,atol=1e-12,rtol=1e-12):
            raise RuntimeError("Synthetic Fourier formula test failed.")
    # Dummy set function f(S) is asymmetric interaction, not additive.
    rows={}
    for mode,r,w,c in CONFIGS:
        factors={"gate":mode!="identity","reflection":r,
                 "weight":w,"bias":c}
        score=(.73+.04*factors["gate"]-.02*factors["weight"]
               +.01*factors["bias"]-.015*(
                   factors["gate"] and factors["reflection"])
               +.027*(factors["gate"] and factors["weight"]))
        rows[name(mode,r,w,c)]={"spearman":float(score)}
    for mode in ("static","dynamic"):
        _=shapley_by_mode(rows,mode)
    print("E8 SELF-TEST passed: FFT identity (odd/even padding) "
          "and Shapley closure.",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path",
        default="/home/data/home/wwr_lumos/models/roberta-base")
    parser.add_argument("--stsb_data_path",default="data/text/stsbenchmark")
    parser.add_argument("--pilot_dir",default=
        "outputs/text/stsb_frozen_trained_mean")
    parser.add_argument("--e7_scores",default=
        "outputs/text/stsb_mean_equiv_e7_calibration_rank/"
        "e7_five_seed_calibration_ranking_audit.npz")
    parser.add_argument("--output",default=
        "outputs/text/stsb_mean_equiv_e8_structural_causality/"
        "e8_five_seed_structural_attribution.json")
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    output=Path(args.output)
    archive=output.with_suffix(".npz")
    if (output.exists() or archive.exists()) and not args.check_only:
        raise FileExistsError(
            f"E8 never overwrites {output} or {archive}")
    sources=preflight(args)
    if not Path(args.e7_scores).is_file():
        raise FileNotFoundError(
            f"Need E7 NPZ prediction baseline: {args.e7_scores}")
    with np.load(args.e7_scores,allow_pickle=False) as npz:
        for seed in SEEDS:
            for split in ("train","validation"):
                for suffix in ("labels","raw_score_matrix"):
                    key=f"seed_{seed}_{split}_{suffix}"
                    if key not in npz:
                        raise RuntimeError(f"Missing E7 cache field {key}")
        if args.check_only:
            print("E8 preflight PASSED: E0 and E7 for seeds 0..4. "
                  "No GPU replay or output.",flush=True)
            return
        device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        tokenizer=AutoTokenizer.from_pretrained(
            args.model_path,local_files_only=True)
        results={}
        full_archive={}
        for seed in SEEDS:
            print(f"\n[E8] seed={seed}: exact E0 10 Adam updates",flush=True)
            result,seed_archive=replay_seed(
                args,seed,tokenizer,sources[seed],device,npz)
            results[str(seed)]=result
            for key,value in seed_archive.items():
                full_archive[f"seed_{seed}_{key}"]=value
    report={
        "experiment":"E8_exact_algebra_structural_attribution_five_seeds",
        "scope":"STSB frozen RoBERTa B2, trained Mean equivalent, "
                "exact E0 Step10, shared original Step10 Head",
        "question":"Are wrong ranking changes attributable to static "
                   "feature rescaling, example-conditioned Gate, "
                   "real/imag reflection, Projection W or bias?",
        "exact_identity":
            "z=W*[a(X)*m+.5*(a(X)-b(X))*(r-m)]+c; "
            "m=valid-masked Mean; "
            "r=valid-masked Mean of circularly reversed masked FFT input; "
            "a,b=2*sigmoid(real/imag Gate), shared across frequency bins.",
        "design":{
            "static_gate":"TRAIN mean of a(X),b(X) over all sentence "
                "occurrences, under E7 batch64 dynamic padding. "
                "No validation information used to set Gate.",
            "counterfactuals":20,
            "factors":"Gate identity/static/dynamic; reflection off/on "
                "(off for identity); projection W identity/trained; "
                "projection c zero/trained.",
            "checks":"Every actual B2 pooled vector must match the "
                "FFT closed form to 2e-5, both splits and five seeds; "
                "same prediction scores as E7 cached Mean/full; "
                "Step0/1/10 exact E0 replay.",
            "Shapley":"Four-factor symmetric allocation of observed "
                "Spearman changes, separately for static and dynamic "
                "Gate; interaction contributions are shared and "
                "not unique causal effects of trained parameters.",
            "rank_audit":"Exact all-example-pairs validation comparison "
                "against Mean and FULL, excluding tied gold labels.",
            "no_new_training":"Only original exact 10-step E0 replay. "
                "No independently refitted Gate, Projection, or Head.",
        },
        "seeds":results,
        "summary":summary(results),
        "limitations":[
            "A frozen checkpoint counterfactual identifies functional "
            "dependence at that checkpoint, not what the original "
            "optimizer would have learned under a different model.",
            "The Gate dynamic vs static contrast also includes "
            "batch dynamic-padding dependence, not solely semantic "
            "content of the sentence. Previous E3c addresses padding.",
            "The learned static Gate is estimated using TRAIN input "
            "distribution; any static model comparison on TRAIN "
            "uses in-sample (unlabeled) gate moments.",
            "Projection and Gate effects interact. One-factor removal "
            "and Shapley allocate effects differently; neither is "
            "a unique causal decomposition of Spearman.",
            "The original Mean was selected on this validation split; "
            "do not claim independent generalization.",
            "Five seeds share STSB and represent Step10 only. "
            "Do not extrapolate to Sprint/AMP or original FLaG.",
        ],
    }
    serial=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n"
    output.parent.mkdir(parents=True,exist_ok=True)
    with archive.open("xb") as f:
        np.savez_compressed(f,**full_archive)
    with output.open("x",encoding="utf-8") as f:f.write(serial)
    print("\nE8 FIVE-SEED STRUCTURAL SUMMARY",flush=True)
    for key in (
        "total_full_minus_mean_spearman",
        "static_minus_dynamic_spearman",
        "full_no_reflection_restoration",
        "full_no_projection_weight_restoration",
        "full_no_bias_restoration",
        "full_static_gate_restoration"):
        s=report["summary"]["validation"][key]
        print(f"  {key}: {s['mean']:+.6f} "
              f"±{s['std_over_seeds']:.6f}",flush=True)
    for f in FACTORS:
        s=report["summary"]["validation"][f"dynamic_shapley_{f}"]
        print(f"  Shapley {f}: {s['mean']:+.6f}",flush=True)
    print(f"Saved exclusive E8 result: {output}",flush=True)
    print(f"Saved 20-way per-pair score archive: {archive}",flush=True)


if __name__=="__main__":
    main()
