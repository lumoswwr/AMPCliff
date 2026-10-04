#!/usr/bin/env python3
"""Probe trained AlignmentAnchorStopGrad geometry on STSB and Sprint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = _REPO_ROOT.parent
for _p in (_REPO_PARENT, _REPO_ROOT / "text_repro"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from train_sts import STSCollator, STSDataset, SentenceEncoder
from train_sprint import SprintCollator, SprintDataset, SprintPairClassifier


def read_json(path):
    with Path(path).open() as f:
        return json.load(f)


def describe(x):
    x=np.asarray(x,dtype=np.float64)
    return (
        float(np.mean(x)),
        float(np.median(x)),
        float(np.percentile(x,10)),
        float(np.percentile(x,90)),
    )


def show(name, alignments, residual_norms, mean_scales, beta):
    a=np.concatenate(alignments)
    r=np.concatenate(residual_norms)
    s=np.concatenate(mean_scales)
    angle=np.degrees(np.arccos(np.clip(a,-1.0,1.0)))
    eff=np.abs(beta)*r

    print()
    print("="*86)
    print(name)
    print("="*86)
    print(f"beta                     {beta:.6f}")

    for label,x in [
        ("alignment <m,f>",a),
        ("angle(m,f) deg",angle),
        ("mean-axis scale 1+a",s),
        ("||r_perp||",r),
        ("|beta|*||r_perp||",eff),
    ]:
        mean,med,p10,p90=describe(x)
        print(
            f"{label:<24s} "
            f"mean={mean:.6f} | median={med:.6f} | "
            f"p10={p10:.6f} | p90={p90:.6f}"
        )


@torch.no_grad()
def collect_pool_geometry(pool, hidden, mask):
    _=pool(hidden,mask)
    a=(
        pool._last_mean_flag_alignment
        .squeeze(-1).detach().cpu().numpy()
    )
    r=(
        pool._last_orthogonal_residual
        .norm(dim=-1).detach().cpu().numpy()
    )
    s=(
        pool._last_mean_axis_scale
        .squeeze(-1).detach().cpu().numpy()
    )
    return a,r,s


@torch.no_grad()
def collect_stsb(seed,device):
    run_dir=Path(
        "outputs/text/stsbenchmark/experiments/"
        f"stsb_unfrozen_flag_alignmentanchor_stopgrad/seed_{seed}"
    )
    cfg=read_json(run_dir/"config.json")
    ckpt=torch.load(run_dir/"best_model.pt",map_location=device)

    tok=AutoTokenizer.from_pretrained(
        cfg["model_path"],local_files_only=True
    )
    raw=load_from_disk(cfg["data_path"])
    loader=DataLoader(
        STSDataset(raw["validation"]),
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=STSCollator(tok,int(cfg["max_length"])),
        num_workers=0,
    )

    model=SentenceEncoder(
        cfg["model_path"],
        "FLaG_AlignmentAnchorStopGrad",
        stft_win_length=int(cfg["stft_win_length"]),
        stft_hop_length=int(cfg["stft_hop_length"]),
        stft_window_type=cfg["stft_window_type"],
        stft_center=bool(cfg["stft_center"]),
        fixed_fft_length=cfg["fixed_fft_length"],
        pool_dropout=float(cfg["pool_dropout"]),
        post_pool_norm=bool(cfg["post_pool_norm"]),
        freeze_backbone=bool(cfg["freeze_backbone"]),
        remove_dc=bool(cfg["remove_dc"]),
        dc_only=bool(cfg["dc_only"]),
        mean_anchor_beta_init=float(cfg["mean_anchor_beta_init"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"],strict=True)
    model.eval()

    beta=float(model.pool.mean_anchor_beta_unbounded.detach().cpu())
    A,R,S=[],[],[]

    for tok1,tok2,_ in loader:
        tok1={k:v.to(device) for k,v in tok1.items()}
        tok2={k:v.to(device) for k,v in tok2.items()}
        combined={
            k:torch.cat([tok1[k],tok2[k]],dim=0)
            for k in tok1 if k in tok2
        }
        hidden=model._backbone_forward(combined).last_hidden_state
        b=tok1["input_ids"].size(0)

        for h,m in [
            (hidden[:b],tok1["attention_mask"]),
            (hidden[b:],tok2["attention_mask"]),
        ]:
            a,r,s=collect_pool_geometry(model.pool,h,m)
            A.append(a); R.append(r); S.append(s)

    return A,R,S,beta


@torch.no_grad()
def collect_sprint(seed,device):
    run_dir=Path(
        "outputs/text/sprintduplicatequestions/experiments/"
        f"sprint_frozen_flag_alignmentanchor_stopgrad/seed_{seed}"
    )
    cfg=read_json(run_dir/"config.json")
    ckpt=torch.load(run_dir/"best_model.pt",map_location=device)

    tok=AutoTokenizer.from_pretrained(
        cfg["model_path"],local_files_only=True
    )
    raw=load_from_disk(cfg["data_path"])
    loader=DataLoader(
        SprintDataset(raw["validation"]),
        batch_size=int(cfg["eval_batch_size"]),
        shuffle=False,
        collate_fn=SprintCollator(tok,int(cfg["max_length"])),
        num_workers=0,
    )

    model=SprintPairClassifier(
        model_path=cfg["model_path"],
        pooling="FLaG_AlignmentAnchorStopGrad",
        stft_win_length=int(cfg["stft_win_length"]),
        stft_hop_length=int(cfg["stft_hop_length"]),
        stft_window_type=cfg["stft_window_type"],
        stft_center=bool(cfg["stft_center"]),
        finetune_backbone=bool(cfg["finetune_backbone"]),
        remove_dc=bool(cfg["remove_dc"]),
        dc_only=bool(cfg["dc_only"]),
        mean_anchor_beta_init=float(cfg["mean_anchor_beta_init"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"],strict=True)
    model.eval()

    beta=float(model.pool.mean_anchor_beta_unbounded.detach().cpu())
    A,R,S=[],[],[]

    for tok1,tok2,_ in loader:
        tok1={k:v.to(device) for k,v in tok1.items()}
        tok2={k:v.to(device) for k,v in tok2.items()}
        combined={
            k:torch.cat([tok1[k],tok2[k]],dim=0)
            for k in tok1 if k in tok2
        }
        hidden=model.backbone(**combined).last_hidden_state
        b=tok1["input_ids"].size(0)

        for h,m in [
            (hidden[:b],tok1["attention_mask"]),
            (hidden[b:],tok2["attention_mask"]),
        ]:
            a,r,s=collect_pool_geometry(model.pool,h,m)
            A.append(a); R.append(r); S.append(s)

    return A,R,S,beta


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args()

    device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:",device)

    A,R,S,b=collect_stsb(args.seed,device)
    show(
        "STSB | validation geometry | AlignmentAnchorStopGrad",
        A,R,S,b,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    A,R,S,b=collect_sprint(args.seed,device)
    show(
        "Sprint | validation geometry | AlignmentAnchorStopGrad",
        A,R,S,b,
    )


if __name__=="__main__":
    main()
