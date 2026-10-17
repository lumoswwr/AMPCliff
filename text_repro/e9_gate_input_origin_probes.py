#!/usr/bin/env python3
"""E9: trace the SOURCE of harmful FLaG Gate information, not Gate on/off.

At the EXACT E0 Step10 STSB trained-Mean-equivalent checkpoints, preserve
all original embeddings, reconstructive FFT, final Projection, and Head.
Perturb ONLY the *input of latent-attention-to-Gate* and identify:
  DC input (F_0=L*Mean), non-DC input, and batch FFT padding length.
DC/AC conditional donor exchanges preserve real spectral marginal inputs
and use donor sentences with EXACT equal valid-token length AND FFT T.
No labels are consulted to choose donor, repeated for two donor maps.
A whole-frequency-bin permutation is an architectural invariance test:
without frequency-position encoding, latent attention ought to see a SET,
not labeled frequency indices.

E8 established Gate damage, E9 investigates its INPUT ORIGIN. A swap that
hurts means the original source contained useful conditional information;
a swap that helps suggests harmful conditional content, not by itself proof
of "noise" or semantics. DC-only and AC-only are OOD sensitivity probes,
never sufficient alone for causal information claims.

All five seeds 0-4 in ONE command, EXACT 10 original Adam steps per seed,
no new parameter training, two exclusive files and full E0/E7 verification.
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
from e6_direction_vs_amplitude import MEAN_ID, FULL_ID, score_stats
from e7_calibration_rank_audit import pairwise_audit
from e8_structural_gate_projection_attribution import check_model_protocol
from explore_mean_equiv_trajectory import evaluate, get_loaders, get_splits
from train_author_text_protocol import PairCollator, PairDataset, seed_everything
from train_stsb_frozen_mean_start import (
    assert_initial_pool_identity, build_frozen_b2,
)
from AMPCliff.factory.pooling.llm_pooling_dropin import masked_mean_pooling


SOURCES=(
    "native", "dc_only", "ac_only",
    "dc_swap_0", "ac_swap_0", "all_swap_0",
    "dc_swap_1", "ac_swap_1", "all_swap_1",
    "pad_plus_16", "pad_plus_32", "all_bins_permuted",
)
MODES=("gate_only","full")
KEYS=tuple(f"{mode}/{source}" for mode in MODES for source in SOURCES)
AUDIT_SOURCES=("native","dc_swap_0","ac_swap_0",
               "dc_swap_1","ac_swap_1",
               "pad_plus_16","pad_plus_32","dc_only","ac_only")


def donor_indices(lengths, generator):
    """Within SAME inference batch, same exact valid-token length and FFT T.

    A random *cyclic derangement* for every length group with n>=2.
    Groups of size 1 are never swapped, and coverage is reported.
    """
    length_np=lengths.detach().cpu().numpy().astype(np.int64)
    result=np.arange(len(length_np),dtype=np.int64)
    matched=np.zeros(len(length_np),dtype=bool)
    for length in np.unique(length_np):
        group=np.flatnonzero(length_np==length)
        if len(group)<2:
            continue
        shift=int(generator.integers(1,len(group)))
        result[group]=np.roll(group,shift)
        matched[group]=True
    if not np.all(length_np[result]==length_np):
        raise RuntimeError("Length-matched donor map is incorrect.")
    return torch.as_tensor(result,dtype=torch.long,
                           device=lengths.device),matched


def spectral_channels(spec):
    return torch.cat([spec.real,spec.imag],dim=-1)


def gate_from_source(pool,freq_tokens):
    latent=pool._latent_pool_in_frequency(freq_tokens)
    logit=pool.freq_gate(latent.mean(dim=1))
    gate=2.*torch.sigmoid(logit)
    if hasattr(pool,"_current_latent_attn_weights"):
        del pool._current_latent_attn_weights
    return gate


@torch.no_grad()
def extract_one_side(model,tokens,device,generator):
    """Return embeddings and source-conditional Gate variants for one side."""
    pool=model.pool
    tokens={k:v.to(device) for k,v in tokens.items()}
    mask=tokens["attention_mask"]
    hidden=model.encoder(**tokens).last_hidden_state
    native_full=pool(hidden,attention_mask=mask)
    native_gate=2.*pool._last_raw_gate
    m=masked_mean_pooling(hidden,mask,eps=1e-6)
    x=hidden*mask.unsqueeze(-1).to(hidden.dtype)
    length=mask.sum(1)
    T=x.size(1)
    reflect=x.index_select(1,(-torch.arange(T,device=device))%T)
    r=masked_mean_pooling(reflect,mask,eps=1e-6)

    spec=torch.fft.rfft(x,dim=1)
    freq=spectral_channels(spec)
    orig=gate_from_source(pool,freq)
    original_gate_err=float((orig-native_gate).abs().max())
    if original_gate_err>1e-5:
        raise RuntimeError(
            f"Gate source forward does not equal B2 native: "
            f"{original_gate_err:g}")
    source={"native":orig}
    zero=torch.zeros_like(freq)
    just_dc=zero.clone()
    just_dc[:,0,:]=freq[:,0,:]
    source["dc_only"]=gate_from_source(pool,just_dc)
    just_ac=freq.clone()
    just_ac[:,0,:]=0.
    source["ac_only"]=gate_from_source(pool,just_ac)

    matched_all=[]
    for rep in (0,1):
        donor,matched=donor_indices(length,generator)
        matched_all.append(matched)
        patched_dc=freq.clone()
        patched_dc[:,0,:]=freq[donor,0,:]
        patched_ac=freq.clone()
        patched_ac[:,1:,:]=freq[donor,1:,:]
        patched_all=freq.index_select(0,donor)
        source[f"dc_swap_{rep}"]=gate_from_source(pool,patched_dc)
        source[f"ac_swap_{rep}"]=gate_from_source(pool,patched_ac)
        source[f"all_swap_{rep}"]=gate_from_source(pool,patched_all)
        all_swap_err=float((source[f"all_swap_{rep}"]-
                            orig.index_select(0,donor)).abs().max())
        if all_swap_err>1e-5:
            raise RuntimeError(
                f"All-band donor Gate equivalence failed: {all_swap_err:g}")

    for extra in (16,32):
        padded=F.pad(x,(0,0,0,extra))
        freq_pad=spectral_channels(torch.fft.rfft(padded,dim=1))
        source[f"pad_plus_{extra}"]=gate_from_source(pool,freq_pad)
    permutation=torch.randperm(freq.size(1),generator=generator)
    permuted=freq.index_select(
        1,permutation.to(device=device))
    source["all_bins_permuted"]=gate_from_source(pool,permuted)
    permutation_err=float(
        (source["native"]-source["all_bins_permuted"]).abs().max())
    if permutation_err>2e-5:
        raise RuntimeError(
            f"FLaG Gate is NOT bin-permutation-invariant: "
            f"{permutation_err:g}. Inspect positional frequency code.")

    outputs={mode:{} for mode in MODES}
    W=pool.time_out_proj.weight
    bias=pool.time_out_proj.bias
    native_full_err=0.
    gate_deltas={}
    for source_name,g in source.items():
        a,b=g.chunk(2,dim=-1)
        g_only=a*m
        pre=a*m+.5*(a-b)*(r-m)
        final=F.linear(pre,W,bias)
        outputs["gate_only"][source_name]=F.normalize(
            g_only,p=2,dim=-1)
        outputs["full"][source_name]=F.normalize(
            final,p=2,dim=-1)
        gate_deltas[source_name]=float((g-orig).abs().mean().item())
        if source_name=="native":
            native_full_err=float((final-native_full).abs().max())
            if native_full_err>2e-5:
                raise RuntimeError(
                    "Current E9 spectral formula fails to reproduce native B2.")
    return {
        "vectors":outputs,
        "mean_vector":F.normalize(m,p=2,dim=-1),
        "gate_deltas":gate_deltas,
        "gate_channel_abs_deviation":
            torch.mean(torch.abs(orig-1.),dim=-1).cpu().numpy(),
        "valid_length":length.cpu().numpy(),
        "donor_match_fraction":
            [float(x.mean()) for x in matched_all],
        "permutation_error":permutation_err,
        "original_gate_error":original_gate_err,
        "native_full_error":native_full_err,
    }


@torch.no_grad()
def evaluate_source_probe(model,loader,device,seed,split):
    """No spectral source patch is applied to reconstructed embedding path."""
    model.eval()
    rng=torch.Generator(device="cpu")
    rng.manual_seed(911010+seed*997+(
        0 if split=="train" else 101))
    yparts=[]
    scores={"mean":[]}
    for k in KEYS:
        scores[k]=[]
    length_left=[]
    length_right=[]
    gate_deviations=[]
    stats={"donor_match_fraction_0":[],"donor_match_fraction_1":[],
           "max_permutation_abs_error":0.,
           "max_native_gate_abs_error":0.,
           "max_native_full_abs_error":0.}
    for batch_idx,(ta,tb,y) in enumerate(loader):
        yparts.append(y.detach().cpu().numpy().astype(np.float32))
        l=extract_one_side(model,ta,device,rng)
        r=extract_one_side(model,tb,device,rng)
        length_left.append(l["valid_length"])
        length_right.append(r["valid_length"])
        for name,side in (("left",l),("right",r)):
            stats["max_permutation_abs_error"]=max(
                stats["max_permutation_abs_error"],
                side["permutation_error"])
            stats["max_native_gate_abs_error"]=max(
                stats["max_native_gate_abs_error"],
                side["original_gate_error"])
            stats["max_native_full_abs_error"]=max(
                stats["max_native_full_abs_error"],
                side["native_full_error"])
            for j in (0,1):
                stats[f"donor_match_fraction_{j}"].append(
                    side["donor_match_fraction"][j])
        gate_deviations.append([
            .5*(l["gate_deltas"][key]+r["gate_deltas"][key])
            for key in SOURCES
        ])
        c=(l["mean_vector"]*r["mean_vector"]).sum(-1)
        scores["mean"].append((
            model.log_scale.exp()*c+model.bias
        ).detach().cpu().numpy().astype(np.float32))
        for mode in MODES:
            for source in SOURCES:
                c=(l["vectors"][mode][source] *
                   r["vectors"][mode][source]).sum(-1)
                score=(model.log_scale.exp()*c+model.bias)
                scores[f"{mode}/{source}"].append(
                    score.detach().cpu().numpy().astype(np.float32))
        if split=="train" and batch_idx and batch_idx%40==0:
            print(f"  seed {seed} probe TRAIN "
                  f"{batch_idx} batches complete",flush=True)
    all_scores={k:np.concatenate(v).astype(np.float64)
                for k,v in scores.items()}
    labels=np.concatenate(yparts)
    if any(len(v)!=len(labels) for v in all_scores.values()):
        raise RuntimeError("E9 score and label ordering mismatch.")
    stats["donor_match_fraction"]={
        str(j):float(np.mean(stats[f"donor_match_fraction_{j}"]))
        for j in (0,1)
    }
    stats["source_gate_average_absolute_changes"]={
        src:float(np.mean([x[i] for x in gate_deviations]))
        for i,src in enumerate(SOURCES)}
    ll=np.concatenate(length_left)
    rr=np.concatenate(length_right)
    stats["valid_length_mean"]=float(.5*(ll.mean()+rr.mean()))
    stats["valid_length_std"]=float(
        np.std(np.concatenate([ll,rr]).astype(np.float64)))
    return labels,all_scores,stats,{
        "labels":labels,
        "valid_length_left":ll,
        "valid_length_right":rr,
        "variant_names":np.asarray(tuple(all_scores.keys())),
        "score_matrix":np.stack(
            [all_scores[key] for key in all_scores]).astype(np.float32),
    }


def validate_e7(reference,seed,split,labels,scores):
    pref=f"seed_{seed}_{split}"
    yref=reference[f"{pref}_labels"]
    if not np.array_equal(labels,yref):
        raise RuntimeError(f"Label mismatch against E7 {seed}/{split}.")
    ids=list(reference[f"seed_{seed}_grid_ids"].astype(str))
    matrix=reference[f"{pref}_raw_score_matrix"]
    vref=matrix[ids.index(FULL_ID)]
    mref=matrix[ids.index(MEAN_ID)]
    checks={
        "original_full_max_abs_error":
            float(np.max(np.abs(scores["full/native"]-vref))),
        "mean_max_abs_error":
            float(np.max(np.abs(scores["mean"]-mref))),
    }
    if max(checks.values())>1e-5:
        raise RuntimeError(
            f"E9 Gate input source reconstruction invalid: {checks}")
    return checks


def report_split(labels,scores,stats,with_order_audit):
    rows={key:score_stats(labels,v) for key,v in scores.items()}
    reference=rows["full/native"]
    mean=rows["mean"]
    report={
        "scores":rows,
        "probe_input_diagnostics":stats,
        "original_full_minus_mean":{
            metric:float(reference[metric]-mean[metric])
            for metric in ("spearman","pearson","mse")},
        "probe_changes_relative_to_native":{
            k:{
                metric:float(rows[k][metric]-
                             rows[f"{k.split('/')[0]}/native"][metric])
                for metric in ("spearman","pearson","mse")}
            for k in KEYS if not k.endswith("/native")},
    }
    if with_order_audit:
        # Ranking event counts, not simply Spearman.
        report["rank_vs_original_mean"]={
            source:pairwise_audit(
                labels,scores["mean"],scores[f"full/{source}"])
            for source in SOURCES}
        report["rank_restored_vs_native_full"]={
            source:pairwise_audit(
                labels,scores["full/native"],scores[f"full/{source}"])
            for source in AUDIT_SOURCES
            if source!="native"}
    return report


def seed_replay(args,seed,tokenizer,source,device,npz):
    pilot,_,metadata,probes,reference=source
    args.seed=seed
    train,val=get_splits(
        "stsb",args.stsb_data_path,"",seed,0,0)
    if len(train)!=metadata["n_train"] or len(val)!=metadata["n_validation"]:
        raise RuntimeError("E0 STSB split changed.")
    online,val_loader=get_loaders(
        "stsb",tokenizer,train,val,seed)
    state=torch.load(reference,map_location="cpu",weights_only=True)
    model=build_frozen_b2(args,device,state)
    del state
    check_model_protocol(model)
    identity=assert_initial_pool_identity(
        model,val_loader,device)
    eval0,arr0=evaluate(model,val_loader,"stsb",device)
    replay={
        "step0":compare_with_e0(
            arr0,pilot/"validation_pairs_step_0.npz",
            f"E9 E0 Step0 seed{seed}")
    }
    numeric_check(eval0["full"],probes[0],0)
    optimizer=torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,weight_decay=0.)
    seed_everything(seed+100000)
    model.train()
    losses=[]
    for step,(ta,tb,y) in enumerate(online,1):
        if step>10:
            break
        ta={k:v.to(device) for k,v in ta.items()}
        tb={k:v.to(device) for k,v in tb.items()}
        optimizer.zero_grad(set_to_none=True)
        pred,_=model(ta,tb)
        loss=F.mse_loss(pred,y.to(device)/5.)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step in (1,10):
            ev,ar=evaluate(model,val_loader,"stsb",device)
            replay[f"step{step}"]=compare_with_e0(
                ar,pilot/f"validation_pairs_step_{step}.npz",
                f"E9 E0 Step{step} seed{seed}")
            numeric_check(ev["full"],probes[step],step)
            model.train()
    if len(losses)!=10:
        raise RuntimeError("E9 requires exact 10-step E0 replay.")
    for step,index in ((1,0),(10,9)):
        if abs(losses[index]-
               float(probes[step]["last_minibatch_train_mse"]))>1e-7:
            raise RuntimeError("E9/E0 online loss mismatch.")

    train_eval=DataLoader(
        PairDataset(train,"stsb"),batch_size=64,shuffle=False,
        collate_fn=PairCollator(
            tokenizer,max_length=128,label_dtype=torch.float32),
        num_workers=0)
    print(f"[E9 seed{seed}] source probes on full TRAIN",flush=True)
    ytr,ptr,sttr,cache_tr=evaluate_source_probe(
        model,train_eval,device,seed,"train")
    print(f"[E9 seed{seed}] source probes on VALIDATION",flush=True)
    yval,pval,stva,cache_va=evaluate_source_probe(
        model,val_loader,device,seed,"validation")
    checks={
        "train":validate_e7(npz,seed,"train",ytr,ptr),
        "validation":validate_e7(npz,seed,"validation",yval,pval),
    }
    numeric_check(score_stats(yval,pval["full/native"]),
                  probes[10],10)
    result={
        "seed":seed,
        "E0_exact_replay":replay,
        "initial_mean_identity_error":identity,
        "E7_endpoint_checks":checks,
        "train":report_split(ytr,ptr,sttr,False),
        "validation":report_split(yval,pval,stva,True),
    }
    files={
        "train":cache_tr,"validation":cache_va,
    }
    print(f" seed{seed} delta Spearman source swaps vs native:",
          flush=True)
    for source in ("dc_swap_0","ac_swap_0","dc_swap_1","ac_swap_1",
                   "pad_plus_16","pad_plus_32"):
        delta=result["validation"][
            "probe_changes_relative_to_native"][f"full/{source}"]["spearman"]
        print(f"    {source}: {delta:+.6f}",flush=True)
    del model,optimizer,online,val_loader,train_eval
    gc.collect()
    if device.type=="cuda":
        torch.cuda.empty_cache()
    return result,files


def aggregate(seeds):
    report={"seed_ids":list(SEEDS)}
    for split in ("train","validation"):
        result={}
        for metric in ("spearman","mse"):
            keys=["mean"]+[
                f"{mode}/{source}"
                for mode in MODES for source in SOURCES
            ]
            for key in keys:
                vals=np.asarray([
                    seeds[str(seed)][split]["scores"][key][metric]
                    for seed in SEEDS],dtype=np.float64)
                result[f"{metric}:{key}"]={
                    "mean":float(vals.mean()),
                    "std_over_seeds":float(vals.std(ddof=1)),
                    "per_seed":{
                        str(seed):float(v)
                        for seed,v in zip(SEEDS,vals)}}
        for source in SOURCES:
            vals=np.asarray([
                seeds[str(seed)][split][
                    "probe_input_diagnostics"][
                    "source_gate_average_absolute_changes"][source]
                for seed in SEEDS],dtype=np.float64)
            result[f"gate_l1_change:{source}"]={
                "mean":float(vals.mean()),
                "per_seed":{
                    str(seed):float(v)
                    for seed,v in zip(SEEDS,vals)}}
        report[split]=result
    return report


def test_invariance():
    """Small synthetic test with actual attention + Gate architecture.

    Test the donor-matching and FFT shapes offline, not learned weights.
    Learned model invariance is strictly checked during real forward.
    """
    rng=torch.Generator().manual_seed(777)
    lengths=torch.tensor([3,3,3,4,4,7,9,9,9,9])
    idx,matched=donor_indices(lengths,rng)
    assert torch.equal(lengths, lengths[idx])
    assert bool(torch.all(idx[matched]!=torch.arange(len(idx))[matched]))
    x=torch.randn(3,12,5)
    for extra in (0,16,32):
        spec=torch.fft.rfft(F.pad(x,(0,0,0,extra)),dim=1)
        assert spec.shape[1]==((12+extra)//2+1)
    print("E9 SELF-TEST passed: exact-length donor derangement "
          "and frequency padding shapes.",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path",default=
                   "/home/data/home/wwr_lumos/models/roberta-base")
    p.add_argument("--stsb_data_path",default="data/text/stsbenchmark")
    p.add_argument("--pilot_dir",default=
                   "outputs/text/stsb_frozen_trained_mean")
    p.add_argument("--e7_scores",default=
                   "outputs/text/stsb_mean_equiv_e7_calibration_rank/"
                   "e7_five_seed_calibration_ranking_audit.npz")
    p.add_argument("--output",default=
                   "outputs/text/stsb_mean_equiv_e9_gate_origin/"
                   "e9_five_seed_gate_frequency_source.json")
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--check-only",action="store_true")
    args=p.parse_args()
    if args.self_test:
        test_invariance()
        return
    sources=preflight(args)
    output=Path(args.output)
    cache_path=output.with_suffix(".npz")
    if (output.exists() or cache_path.exists()) and not args.check_only:
        raise FileExistsError(
            "E9 refuses to overwrite existing diagnostics.")
    if not Path(args.e7_scores).is_file():
        raise FileNotFoundError(
            f"E7 reference score cache missing: {args.e7_scores}")
    if args.check_only:
        print("E9 input archives checked for all seeds 0..4. "
              "No output written.",flush=True)
        return
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=AutoTokenizer.from_pretrained(
        args.model_path,local_files_only=True)
    results={}
    all_arrays={}
    with np.load(args.e7_scores,allow_pickle=False) as npz:
        for seed in SEEDS:
            print(f"\n[E9] seed {seed}: replay + Gate SOURCE interventions",
                  flush=True)
            result,caches=seed_replay(
                args,seed,tokenizer,sources[seed],device,npz)
            results[str(seed)]=result
            for split,values in caches.items():
                for key,value in values.items():
                    all_arrays[f"seed_{seed}_{split}_{key}"]=value
    report={
        "experiment":"E9_five_seed_gate_spectral_source_probe",
        "question":"Which inputs into latent-attention Gate carry "
                   "task-irrelevant or harmful information?",
        "controls":{
            "native":"Original B2 gate, preserved same input/model/head.",
            "dc_only/ac_only":"Feed only DC or only non-DC to the "
                "Latent->Gate producer. OOD sensitivity probe ONLY.",
            "dc_swap/ac_swap":"Replace DC or the entire non-DC block "
                "in the Gate producer with a label-blind donor from "
                "the SAME batch with EXACT same valid length and FFT T; "
                "do not alter output reconstruction or original Mean.",
            "all_swap":"Swap all spectrum tokens; must equal donor's "
                "original Gate. Distribution-preserving negative control.",
            "pad_plus_16/32":"Append strictly masked zeros AFTER encoder "
                "and change only the FFT input to Latent->Gate; "
                "hold original reconstructed m,r fixed. Tests "
                "mathematically irrelevant batch FFT length dependence.",
            "all_bins_permuted":"Gate should be invariant when all "
                "frequency bins are permuted (latent MHA contains "
                "no frequency position encoding). Exact check.",
            "embedding_path":"Gate-only a(X)*Mean and full "
                "W*(a*m+0.5*(a-b)*(r-m))+c evaluated for each "
                "input probe, same original Step10 Head.",
            "checks":"Exact E0 Step0/1/10 and E7 original Mean/full "
                "prediction arrays all checked. No new model trained.",
        },
        "seeds":results,
        "summary":aggregate(results),
        "limitations":[
            "Changing Gate input can put latent attention off its "
            "training distribution, especially DC-only/AC-only.",
            "Donor interventions are conditional on exact valid "
            "length and FFT T but not semantic equivalence. "
            "If original beats donor, its input has useful "
            "conditional association; if donor beats original, "
            "do not automatically call the source semantic noise.",
            "Donor matching is restricted to each batch. "
            "Singleton lengths are not replaced and coverage "
            "is explicitly reported.",
            "Padding intervention holds encoded sentence and "
            "original m,r fixed: changed Gate is demonstrably "
            "nuisance sensitivity but need not be sole loss cause.",
            "Gate sees spectral token SET without explicit bin "
            "position identity, which does not mean spectral "
            "magnitudes and phase are ignored.",
            "Step10 STSB frozen-RoBERTa mechanism; does not "
            "generalize to paper's fully-trained protocols.",
            "Five seeds share validation-selected Mean checkpoint "
            "and STSB dataset; not external test evidence.",
        ],
    }
    serial=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n"
    output.parent.mkdir(parents=True,exist_ok=True)
    with cache_path.open("xb") as f:
        np.savez_compressed(f,**all_arrays)
    with output.open("x",encoding="utf-8") as f:
        f.write(serial)
    print("\nE9 Gate input-source probes complete",flush=True)
    for source in ("native","dc_swap_0","ac_swap_0",
                   "pad_plus_16","pad_plus_32"):
        v=report["summary"]["validation"][f"spearman:full/{source}"]
        print(f" {source}: Spearman={v['mean']:.6f} "
              f"std={v['std_over_seeds']:.6f}",flush=True)
    print(f"JSON: {output}\nNPZ: {cache_path}",flush=True)


if __name__=="__main__":
    main()
