#!/usr/bin/env python3
"""Probe Mean/FLaG geometry for trained unbounded Mean-anchor checkpoints."""

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
from train_sprint import (
    SprintCollator,
    SprintDataset,
    SprintPairClassifier,
)

def read_json(path):
    with Path(path).open() as f:
        return json.load(f)

def summarize(name, alignments, residual_norms, beta):
    a=np.concatenate(alignments)
    r=np.concatenate(residual_norms)
    eff=np.abs(beta)*r
    angle=np.degrees(np.arccos(np.clip(a,-1.0,1.0)))

    def stats(x):
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "p10": float(np.percentile(x,10)),
            "p90": float(np.percentile(x,90)),
        }

    print()
    print("="*82)
    print(name)
    print("="*82)
    print(f"beta                     {beta:.6f}")
    for label,x in [
        ("alignment <m,f>",a),
        ("angle(m,f) deg",angle),
        ("||r_perp||",r),
        ("|beta|*||r_perp||",eff),
    ]:
        s=stats(x)
        print(
            f"{label:<24s} "
            f"mean={s['mean']:.6f} | "
            f"median={s['median']:.6f} | "
            f"p10={s['p10']:.6f} | "
            f"p90={s['p90']:.6f}"
        )

@torch.no_grad()
def collect_stsb(seed, device):
    run_dir=Path(
        "outputs/text/stsbenchmark/experiments/"
        f"stsb_unfrozen_flag_meananchor_free/seed_{seed}"
    )
    cfg=read_json(run_dir/"config.json")
    ckpt=torch.load(run_dir/"best_model.pt",map_location=device)

    tok=AutoTokenizer.from_pretrained(
        cfg["model_path"], local_files_only=True
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
        "FLaG_MeanAnchorFree",
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
    aligns=[]
    norms=[]

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
            _=model.pool(h,m)
            aligns.append(
                model.pool._last_mean_flag_alignment
                .squeeze(-1).detach().cpu().numpy()
            )
            norms.append(
                model.pool._last_orthogonal_residual
                .norm(dim=-1).detach().cpu().numpy()
            )

    return aligns,norms,beta

@torch.no_grad()
def collect_sprint(seed, device):
    run_dir=Path(
        "outputs/text/sprintduplicatequestions/experiments/"
        f"sprint_frozen_flag_meananchor_free/seed_{seed}"
    )
    cfg=read_json(run_dir/"config.json")
    ckpt=torch.load(run_dir/"best_model.pt",map_location=device)

    tok=AutoTokenizer.from_pretrained(
        cfg["model_path"], local_files_only=True
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
        pooling="FLaG_MeanAnchorFree",
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
    aligns=[]
    norms=[]

    for tok1,tok2,_ in loader:
        tok1={k:v.to(device) for k,v in tok1.items()}
        tok2={k:v.to(device) for k,v in tok2.items()}
        combined={
            k:torch.cat([tok1[k],tok2[k]],dim=0)
            for k in tok1 if k in tok2
        }

        if model.encoder.finetune_backbone:
            hidden=model.backbone(**combined).last_hidden_state
        else:
            hidden=model.backbone(**combined).last_hidden_state

        b=tok1["input_ids"].size(0)

        for h,m in [
            (hidden[:b],tok1["attention_mask"]),
            (hidden[b:],tok2["attention_mask"]),
        ]:
            _=model.pool(h,m)
            aligns.append(
                model.pool._last_mean_flag_alignment
                .squeeze(-1).detach().cpu().numpy()
            )
            norms.append(
                model.pool._last_orthogonal_residual
                .norm(dim=-1).detach().cpu().numpy()
            )

    return aligns,norms,beta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=0)
    args=ap.parse_args()

    device=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:",device)

    a,r,b=collect_stsb(args.seed,device)
    summarize(
        "STSB | validation geometry | MeanAnchorFree",
        a,r,b,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    a,r,b=collect_sprint(args.seed,device)
    summarize(
        "Sprint | validation geometry | MeanAnchorFree",
        a,r,b,
    )

if __name__=="__main__":
    main()
