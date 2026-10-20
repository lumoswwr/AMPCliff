#!/usr/bin/env python3
"""STSB: compare standard MSE and RankNet pairwise loss for Mean, FLaG-Zero, FLaG.

Runs 3 architectures x 2 objectives x seeds 0..4. Outputs are stable and
completed runs are never overwritten. Checkpoint selection uses validation
Spearman; test metrics are computed only after selection.
"""
import argparse, csv, gc, json, math, statistics, sys
from pathlib import Path

# Ensure imports such as `from AMPCliff.factory...` work when launched from
# inside the repository checkout (e.g. `python text_repro/script.py`).
# This adds the checkout's parent directory to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_author_text_protocol import (
    PairCollator, PairDataset, SentencePairModel, build_e2e_adam, seed_everything
)

MODELS = ("mean", "flag_zero", "flag")
LOSSES = ("mse", "ranknet")
SEEDS = range(5)


def ranknet_loss(cosine, labels):
    """RankNet logistic loss over all non-tied unordered examples in a minibatch."""
    n = labels.numel()
    if n < 2:
        return cosine.sum() * 0.0, 0
    ij = torch.triu_indices(n, n, offset=1, device=labels.device)
    i, j = ij[0], ij[1]
    dy = labels[i] - labels[j]
    valid = dy.abs() > 1e-8
    count = int(valid.sum().item())
    if not count:
        return cosine.sum() * 0.0, 0
    ds = cosine[i[valid]] - cosine[j[valid]]
    return F.softplus(-torch.sign(dy[valid]) * ds).mean(), count


def corr(fn, y, p):
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    v = fn(y, p).statistic
    return float(v) if np.isfinite(v) else float("nan")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, scores, cosines = [], [], []
    for t1, t2, labels in loader:
        t1 = {k: v.to(device) for k, v in t1.items()}
        t2 = {k: v.to(device) for k, v in t2.items()}
        score, cosine = model(t1, t2)
        ys.extend(labels.numpy().tolist())
        scores.extend(score.cpu().numpy().tolist())
        cosines.extend(cosine.cpu().numpy().tolist())
    y = np.asarray(ys, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    c = np.asarray(cosines, dtype=np.float64)
    return {
        "n_examples": int(len(y)),
        "spearman": corr(spearmanr, y, c),
        "pearson": corr(pearsonr, y, c),
        "native_mse": float(np.mean((s-y/5.0)**2)),
    }, {"labels": y, "scores": s, "cosine": c}


def fit_positive_affine(cosine, labels):
    """Fit y ~= a*cosine+b on train only; constrain a to be positive."""
    x = np.asarray(cosine, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)/5.0
    xc = x-x.mean()
    denom = float(np.dot(xc, xc))
    a = 1e-8 if denom <= 1e-12 else float(np.dot(xc, y-y.mean())/denom)
    a = max(a, 1e-8)
    return a, float(y.mean()-a*x.mean())


def make_model(name, args, device):
    if name == "mean":
        pooling, time_pool, post_norm = "mean", "max", True
    elif name == "flag_zero":
        pooling, time_pool, post_norm = "FLaG_B2", "mean", False
    else:
        pooling, time_pool, post_norm = "FLaG", "max", True
    return SentencePairModel(
        args.model_path, pooling, freeze_backbone=False,
        mean_anchor_beta_init=0.1, flag_time_pool=time_pool,
        flag_post_pool_norm=post_norm
    ).to(device)


def check_flag_zero(model, batch, device):
    t1, t2, _ = batch
    t1 = {k:v.to(device) for k,v in t1.items()}
    t2 = {k:v.to(device) for k,v in t2.items()}
    model.eval()
    with torch.no_grad():
        _, actual = model(t1, t2)
        h1 = model.encoder(**t1).last_hidden_state
        h2 = model.encoder(**t2).last_hidden_state
        def mean_pool(h, mask):
            m = mask.unsqueeze(-1).to(h.dtype)
            return (h*m).sum(1)/m.sum(1).clamp(min=1e-6)
        p1 = model.pool(h1, t1["attention_mask"])
        p2 = model.pool(h2, t2["attention_mask"])
        m1 = mean_pool(h1, t1["attention_mask"])
        m2 = mean_pool(h2, t2["attention_mask"])
        pd = max(float((p1-m1).abs().max().cpu()), float((p2-m2).abs().max().cpu()))
        expected = (F.normalize(m1,dim=-1)*F.normalize(m2,dim=-1)).sum(-1)
        cd = float((actual-expected).abs().max().cpu())
    model.train()
    if not math.isfinite(pd+cd) or max(pd,cd)>1e-5:
        raise RuntimeError(f"FLaG-Zero Mean-equivalence check failed: pool={pd:.3e}, cosine={cd:.3e}")
    return {"max_abs_pool_difference":pd,"max_abs_cosine_difference":cd,"tolerance":1e-5}


def run_one(root, name, loss_name, seed, args, raw, tokenizer, device):
    out = root/name/loss_name/f"seed_{seed}"
    metrics_file = out/"metrics.json"
    if metrics_file.exists():
        print(f"[skip completed] {name}/{loss_name}/seed_{seed}", flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite incomplete result directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    collate = PairCollator(tokenizer, max_length=128, label_dtype=torch.float32)
    train_ds, val_ds, test_ds = (PairDataset(raw[k],"stsb") for k in ("train","validation","test"))
    gen = torch.Generator().manual_seed(seed)
    train = DataLoader(train_ds,batch_size=32,shuffle=True,generator=gen,collate_fn=collate,num_workers=0)
    train_eval = DataLoader(train_ds,batch_size=64,shuffle=False,collate_fn=collate,num_workers=0)
    val = DataLoader(val_ds,batch_size=64,shuffle=False,collate_fn=collate,num_workers=0)
    test = DataLoader(test_ds,batch_size=64,shuffle=False,collate_fn=collate,num_workers=0)
    model = make_model(name,args,device)
    init_check = check_flag_zero(model,next(iter(train_eval)),device) if name=="flag_zero" else {"applicable":False}
    optimizer = build_e2e_adam(model,model.encoder,1e-5,1e-3)
    print(f"\n=== {name} | {loss_name} | seed={seed} | device={device} ===",flush=True)
    print(f"split_sizes={len(train_ds)}/{len(val_ds)}/{len(test_ds)} init_check={init_check}",flush=True)
    best_sp, best_epoch = -float("inf"), None
    best_file = out/"best_model.pt"
    history=[]
    for epoch in range(1,4):
        model.train()
        total, weight_sum, pair_sum = 0.0, 0, 0
        for t1,t2,labels in train:
            t1={k:v.to(device) for k,v in t1.items()}
            t2={k:v.to(device) for k,v in t2.items()}
            labels=labels.to(device)
            optimizer.zero_grad()
            pred,cosine=model(t1,t2)
            if loss_name=="mse":
                loss=F.mse_loss(pred,labels/5.0)
                weight=int(labels.numel())
            else:
                loss,weight=ranknet_loss(cosine,labels)
                pair_sum+=weight
                if weight==0:
                    continue
            loss.backward()
            optimizer.step()
            total+=float(loss.detach().cpu())*weight
            weight_sum+=weight
        train_loss=total/max(weight_sum,1)
        vm,_=evaluate(model,val,device)
        history.append({"epoch":epoch,"train_loss":train_loss,"loss":loss_name,
                        "effective_pairs":pair_sum if loss_name=="ranknet" else 0,
                        "val_spearman":vm["spearman"],"val_pearson":vm["pearson"],
                        "val_native_mse":vm["native_mse"]})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_sp={vm['spearman']:.6f} "
              f"val_pe={vm['pearson']:.6f} val_mse={vm['native_mse']:.6f} pairs={pair_sum}",flush=True)
        if np.isfinite(vm["spearman"]) and vm["spearman"]>best_sp:
            best_sp,best_epoch=float(vm["spearman"]),epoch
            torch.save(model.state_dict(),best_file)
    if best_epoch is None:
        raise RuntimeError("No valid validation Spearman; inspect this run before retrying.")
    model.load_state_dict(torch.load(best_file,map_location=device))
    tm,ta=evaluate(model,train_eval,device)
    vm,_=evaluate(model,val,device)
    sm,sa=evaluate(model,test,device)
    slope,intercept=fit_positive_affine(ta["cosine"],ta["labels"])
    calibrated=slope*sa["cosine"]+intercept
    result={
        "model":name,"loss":loss_name,"seed":seed,"best_epoch":best_epoch,
        "checkpoint_selection":"validation Spearman",
        "train_size":len(train_ds),"validation_size":len(val_ds),"test_size":len(test_ds),
        "train_spearman":tm["spearman"],"best_val_spearman":vm["spearman"],
        "best_val_pearson":vm["pearson"],"val_native_mse":vm["native_mse"],
        "test_spearman":sm["spearman"],"test_pearson":sm["pearson"],
        "test_native_mse":sm["native_mse"],
        "test_train_calibrated_mse":float(np.mean((calibrated-sa["labels"]/5.0)**2)),
        "train_fit_positive_slope":slope,"train_fit_intercept":intercept,
        "ranknet_definition":"mean softplus(-sign(y_i-y_j)*(cos_i-cos_j)) over all non-tied unordered examples in each minibatch",
        "optimizer":"Adam","backbone_lr":1e-5,"pool_and_head_lr":1e-3,
        "weight_decay":0.0,"epochs":3,"train_batch_size":32,"eval_batch_size":64,"max_length":128,
        "initial_equivalence_check":init_check,
        "native_mse_note":"RankNet optimizes raw cosine and does not train the affine score head; use test_train_calibrated_mse for cross-loss MSE comparison."
    }
    fields=sorted({k for row in history for k in row})
    with (out/"history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(history)
    with metrics_file.open("w") as f: json.dump(result,f,indent=2)
    print("RESULT "+json.dumps(result),flush=True)
    del model,optimizer,train,train_eval,val,test
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def mean_sd(xs):
    return statistics.mean(xs), statistics.stdev(xs) if len(xs)>1 else 0.0


def summarize(root):
    rec={}
    missing=[]
    for m in MODELS:
        for loss in LOSSES:
            for s in SEEDS:
                p=root/m/loss/f"seed_{s}"/"metrics.json"
                if not p.exists(): missing.append(str(p))
                else: rec[(m,loss,s)]=json.loads(p.read_text())
    if missing: raise FileNotFoundError("Missing completed runs:\n"+"\n".join(missing))
    metrics=("test_spearman","test_pearson","test_train_calibrated_mse","train_spearman","best_val_spearman")
    rows=[]
    for m in MODELS:
        for loss in LOSSES:
            row={"model":m,"loss":loss,"n_seeds":5}
            for metric in metrics:
                a,b=mean_sd([rec[(m,loss,s)][metric] for s in SEEDS])
                row[metric+"_mean"],row[metric+"_sd"]=a,b
            rows.append(row)
    with (root/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    deltas=[]
    for m in MODELS:
        for metric in ("test_spearman","test_train_calibrated_mse"):
            ds=[rec[(m,"ranknet",s)][metric]-rec[(m,"mse",s)][metric] for s in SEEDS]
            a,b=mean_sd(ds);deltas.append({"comparison":m+":ranknet_minus_mse","metric":metric,"mean_delta":a,"sd_delta":b,"per_seed_deltas":json.dumps(ds)})
    for loss in LOSSES:
        for m in ("flag_zero","flag"):
            for metric in ("test_spearman","test_train_calibrated_mse"):
                ds=[rec[(m,loss,s)][metric]-rec[("mean",loss,s)][metric] for s in SEEDS]
                a,b=mean_sd(ds);deltas.append({"comparison":m+"_minus_mean_under_"+loss,"metric":metric,"mean_delta":a,"sd_delta":b,"per_seed_deltas":json.dumps(ds)})
    with (root/"paired_differences.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(deltas[0]));w.writeheader();w.writerows(deltas)
    lines=["# STSB: MSE vs standard RankNet","",
           "Mean ± sample SD across seeds 0–4. Checkpoints selected by validation Spearman.",
           "Calibrated MSE fits a positive-slope affine map on training data only, then evaluates on test.","",
           "| Model | Loss | Test Spearman | Test Pearson | Train-calibrated test MSE |",
           "|---|---|---:|---:|---:|"]
    for r in rows:
        def fmt(k): return f'{r[k+"_mean"]:.6f} ± {r[k+"_sd"]:.6f}'
        lines.append(f'| {r["model"]} | {r["loss"]} | {fmt("test_spearman")} | {fmt("test_pearson")} | {fmt("test_train_calibrated_mse")} |')
    lines+=["","## Paired differences","","Positive Spearman delta is better; negative calibrated-MSE delta is better.","",
            "| Comparison | Metric | Mean delta | SD | Per-seed deltas |","|---|---|---:|---:|---|"]
    for r in deltas: lines.append(f'| {r["comparison"]} | {r["metric"]} | {r["mean_delta"]:.6f} | {r["sd_delta"]:.6f} | {r["per_seed_deltas"]} |')
    (root/"summary.md").write_text("\n".join(lines)+"\n")
    print((root/"summary.md").read_text())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output_root",default="outputs/text/stsb_ltr_ranknet_standard")
    ap.add_argument("--model_path",default="/home/data/home/wwr_lumos/models/roberta-base")
    ap.add_argument("--stsb_data_path",default="data/text/stsbenchmark")
    ap.add_argument("--summary_only",action="store_true")
    args=ap.parse_args()
    root=Path(args.output_root);root.mkdir(parents=True,exist_ok=True)
    if args.summary_only: summarize(root);return
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw=load_from_disk(args.stsb_data_path)
    tokenizer=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True)
    print(f"Device={device}; output={root}; seeds=0..4",flush=True)
    for m in MODELS:
        for loss in LOSSES:
            for seed in SEEDS: run_one(root,m,loss,seed,args,raw,tokenizer,device)
    summarize(root)


if __name__=="__main__":
    main()
