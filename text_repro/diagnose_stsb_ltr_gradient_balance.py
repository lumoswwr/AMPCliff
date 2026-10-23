#!/usr/bin/env python3
"""Read-only gradient audit at a trained STSB Mean-MSE checkpoint.

Computes cosine similarity between MSE and rank-loss gradients on identical
forward passes, without optimizer steps, saving checkpoints, or evaluating test.
Examines fixed margin 1, fixed margin 0.1 and label-proportional margin.

Run:
  python -u text_repro/diagnose_stsb_ltr_gradient_balance.py
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_author_text_protocol import (
    PairCollator, PairDataset, seed_everything,
)
from run_stsb_ranknet_comparison import make_model


def grad_snapshot(model):
    """Capture one gradient vector per parameter on CPU, avoiding extra GPU copies."""
    return {
        name: p.grad.detach().float().cpu().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


def grad_compare(mse_grads, model):
    groups = {"encoder": [0.0, 0.0, 0.0],
              "pool": [0.0, 0.0, 0.0],
              "score_head": [0.0, 0.0, 0.0],
              "all": [0.0, 0.0, 0.0]}
    for name, p in model.named_parameters():
        group = ("encoder" if name.startswith("encoder.") else
                 "pool" if name.startswith("pool.") else "score_head")
        g1 = mse_grads.get(name)
        g2 = p.grad.detach().float().cpu() if p.grad is not None else None
        if g1 is None and g2 is None:
            continue
        d1 = float((g1.double().square().sum()).item()) if g1 is not None else 0.0
        d2 = float((g2.double().square().sum()).item()) if g2 is not None else 0.0
        dot = float((g1.double() * g2.double()).sum().item()) if g1 is not None and g2 is not None else 0.0
        for dest in (group, "all"):
            groups[dest][0] += d1
            groups[dest][1] += d2
            groups[dest][2] += dot

    result = {}
    for name, (ms, rs, dot) in groups.items():
        m_norm, r_norm = math.sqrt(ms), math.sqrt(rs)
        result[name] = {
            "mse_norm": m_norm, "rank_norm": r_norm,
            "rank_to_mse_norm_ratio": r_norm / max(m_norm, 1e-15),
            "weighted_ratio_w005": (0.05 * r_norm) / max(m_norm, 1e-15),
            "cosine": dot / (m_norm * r_norm) if m_norm > 0 and r_norm > 0 else float("nan"),
        }
    return result


def ranking_hinge(scores, labels, *, mode, near_tie_threshold):
    n = labels.numel()
    ij = torch.triu_indices(n, n, offset=1, device=labels.device)
    i, j = ij.unbind(0)
    delta = labels[i] - labels[j]
    mask = delta.abs() > 1e-8
    if mode == "gap_filter":
        mask = mask & (delta.abs() >= near_tie_threshold)
    n_pairs = int(mask.sum().item())
    if n_pairs == 0:
        return scores.sum() * 0.0, n_pairs, 0.0
    dy = delta[mask]
    signed = torch.sign(dy) * (scores[i[mask]] - scores[j[mask]])
    if mode == "margin1" or mode == "gap_filter":
        margin = torch.ones_like(signed)
    elif mode == "margin01":
        margin = torch.full_like(signed, 0.1)
    elif mode == "label_margin":
        margin = dy.abs() / 5.0
    else:
        raise ValueError(mode)
    return F.relu(margin - signed).mean(), n_pairs, float((signed < margin).float().mean().item())


def load_source(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected state_dict, got {type(obj)}")
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/home/data/home/wwr_lumos/models/roberta-base")
    p.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    p.add_argument("--checkpoint_root", default="outputs/text/stsb_ltr_v2_e20_min12_pat5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--batches", type=int, default=2)
    p.add_argument("--near_tie_threshold", type=float, default=0.5)
    args = p.parse_args()
    if args.batch_size < 2 or args.batches < 1:
        p.error("batch_size >= 2 and batches >= 1 required")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = (Path(args.checkpoint_root) / "mean" / "mse" /
              f"seed_{args.seed}" / "best_model.pt")
    model = make_model("mean", args, device)
    model.load_state_dict(load_source(source), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    raw = load_from_disk(args.stsb_data_path)
    loader = DataLoader(
        PairDataset(raw["train"], "stsb"),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=PairCollator(tokenizer, 128, torch.float32),
        num_workers=0,
    )

    names = ("margin1", "margin01", "label_margin", "gap_filter")
    print(f"[setup] source={source} device={device} seed={args.seed} "
          f"batch_size={args.batch_size} batches={args.batches}", flush=True)
    print("No optimizer step, no test evaluation, no training-file writes.", flush=True)
    print("Reported weighted_ratio is w=0.05 * ||ranking grad|| / ||MSE grad||.", flush=True)
    model.train()
    for batch_i, (t1, t2, labels) in enumerate(loader):
        if batch_i >= args.batches:
            break
        t1 = {k: v.to(device) for k, v in t1.items()}
        t2 = {k: v.to(device) for k, v in t2.items()}
        labels = labels.to(device)
        model.zero_grad(set_to_none=True)
        pred, cosine = model(t1, t2)
        loss_mse = F.mse_loss(pred, labels / 5.0)
        ranking = {
            name: ranking_hinge(pred, labels, mode=name,
                                near_tie_threshold=args.near_tie_threshold)
            for name in names
        }

        # Every branch shares EXACTLY the same forward pass, so dropout does
        # not invalidate the gradient-direction comparison.
        loss_mse.backward(retain_graph=True)
        baseline_grads = grad_snapshot(model)
        print(f"\n[BATCH {batch_i}] mse={loss_mse.item():.6f} "
              f"scale={model.log_scale.exp().item():.6f}", flush=True)
        for pos, (name, (loss, n_pairs, violation)) in enumerate(ranking.items()):
            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=(pos < len(ranking) - 1))
            stat = grad_compare(baseline_grads, model)
            enc = stat["encoder"]
            head = stat["score_head"]
            print(
                f"  {name:13} loss={loss.item():.6f} pairs={n_pairs:3} "
                f"viol={violation:.3f} "
                f"encoder: mse_norm={enc['mse_norm']:.5f} "
                f"rank_norm={enc['rank_norm']:.5f} "
                f"cos={enc['cosine']:.4f} "
                f"weighted_ratio={enc['weighted_ratio_w005']:.4f}; "
                f"score_head: cos={head['cosine']:.4f} "
                f"weighted_ratio={head['weighted_ratio_w005']:.4f}",
                flush=True,
            )
        model.zero_grad(set_to_none=True)
        del baseline_grads, ranking, pred, cosine, loss_mse
    print("\n[complete] Read-only gradient audit finished.", flush=True)


if __name__ == "__main__":
    main()
