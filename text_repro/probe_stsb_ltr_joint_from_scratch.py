#!/usr/bin/env python3
"""Focused STSB joint training from RoBERTa-base (NO TEST EVALUATION).

Four matched objectives from the same pretrained encoder, same initialization,
shuffle seed, optimizer, learning rates and validation-selection protocol.
Isolates training-stage timing from warm-start. Independent output paths.
Only validation Spearman selects checkpoints, including epoch 0.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from train_author_text_protocol import (
    PairCollator, PairDataset, build_e2e_adam, seed_everything,
)
from run_stsb_ranknet_comparison import (
    check_flag_zero, evaluate, make_model,
)

VARIANTS = ("mse_scratch", "mse_hinge_margin01", "mse_hinge_labelmargin", "mse_listnet")
MODELS = ("mean",)


def gap_hinge(scores, labels, min_gap):
    """Label-gap filtering; return loss and selected/count ratio for diagnostics."""
    i, j = torch.triu_indices(labels.numel(), labels.numel(), 1, device=labels.device)
    dy = labels[i] - labels[j]
    valid = (dy.abs() > 1e-8) & (dy.abs() >= min_gap)
    count = int(valid.sum().item())
    if not count:
        return scores.sum() * 0.0, 0, int(i.numel())
    signed = torch.sign(dy[valid]) * (scores[i[valid]] - scores[j[valid]])
    return F.relu(1.0 - signed).mean(), count, int(i.numel())


def calibrated_hinge(scores, labels, margin_mode):
    """Fixed 0.1 or label-distance/5 Hinge on non-tied pairs.

    margin_mode='fixed01'      margin=0.1
    margin_mode='label_margin' margin=abs(label_i-label_j)/5
    Both variants retain exactly the v2 pair aggregation / sign convention.
    """
    i, j = torch.triu_indices(labels.numel(), labels.numel(), 1, device=labels.device)
    dy = labels[i] - labels[j]
    valid = dy.abs() > 1e-8
    count = int(valid.sum().item())
    if not count:
        return scores.sum() * 0.0, 0, int(i.numel())
    delta = dy[valid]
    signed = torch.sign(delta) * (scores[i[valid]] - scores[j[valid]])
    if margin_mode == "fixed01":
        margin = torch.full_like(signed, 0.1)
    elif margin_mode == "label_margin":
        margin = delta.abs() / 5.0
    else:
        raise ValueError(f"unknown margin_mode: {margin_mode}")
    return F.relu(margin - signed).mean(), count, int(i.numel())


def listnet_kl(scores, labels):
    """Numerically stable ListNet KL with BOTH target and score on [0,1] label scale.

    Unlike the old ListNet, this compares softmax(labels/5) with softmax(scores)
    and removes constant target entropy. This is a distinct, named variant.
    """
    logp = F.log_softmax(scores, dim=0)
    logq = F.log_softmax(labels / 5.0, dim=0)
    q = logq.exp()
    return torch.sum(q * (logq - logp))


def self_test():
    labels = torch.tensor([0.0, 0.1, 1.0, 5.0])
    correct = torch.tensor([0.0, 0.01, 0.8, 4.0], requires_grad=True)
    reversed_ = torch.tensor([4.0, 3.0, 1.0, 0.0], requires_grad=True)
    a, count_a, _ = gap_hinge(correct, labels, 0.5)
    b, count_b, _ = gap_hinge(reversed_, labels, 0.5)
    assert count_a == count_b == 5, (count_a, count_b)
    assert float(a.detach()) < float(b.detach())
    (a + listnet_kl(correct, labels)).backward()
    assert correct.grad is not None and torch.isfinite(correct.grad).all().item()
    assert abs(float(listnet_kl(labels / 5.0, labels).detach())) < 1e-5
    for mode in ("fixed01", "label_margin"):
        left, nleft, _ = calibrated_hinge(correct, labels, mode)
        right, nright, _ = calibrated_hinge(reversed_, labels, mode)
        assert nleft == nright == 6
        assert torch.isfinite(left).item() and torch.isfinite(right).item()
        assert float(left.detach()) < float(right.detach())
        left.backward(retain_graph=True)
        assert correct.grad is not None and torch.isfinite(correct.grad).all().item()
    print("[self-test] gap Hinge, calibrated Hinge, and ListNet KL PASS", flush=True)


def load_checkpoint(model, path, name):
    if not path.is_file():
        raise FileNotFoundError("Source Mean-MSE checkpoint missing: " + str(path))
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older installed PyTorch versions lack the weights_only option.
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError("Expected a raw state_dict at " + str(path))
    load = model.load_state_dict(state, strict=False)
    missing = list(load.missing_keys)
    unexpected = list(load.unexpected_keys)
    if name == "mean":
        if missing or unexpected:
            raise RuntimeError(f"Mean checkpoint incompatible: missing={missing}, unexpected={unexpected}")
    else:
        if unexpected or any(not key.startswith("pool.") for key in missing):
            raise RuntimeError(f"FLaG-Zero warm-start incompatible: missing={missing}, unexpected={unexpected}")
        if not missing:
            raise RuntimeError("FLaG-Zero warm-start did not report newly initialized pool parameters")
    return {"source": str(path), "missing_new_pool_parameters": missing, "unexpected": unexpected}


def make_loader(ds, tokenizer, bs, shuffle=False, seed=0):
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=PairCollator(tokenizer, 128, torch.float32),
        num_workers=0,
    )


def write_history(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one(args, raw, tokenizer, seed, variant, device):
    name = args.model
    target = args.output_root / name / variant / f"seed_{seed}"
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prior diagnostic output: {target}")
    target.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    train_ds = PairDataset(raw["train"], "stsb")
    val_ds = PairDataset(raw["validation"], "stsb")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_ds), size=min(512, len(train_ds)), replace=False)
    train_sample = Subset(train_ds, sorted(int(x) for x in idx))
    loader = make_loader(train_ds, tokenizer, 32, shuffle=True, seed=seed)
    train_eval = make_loader(train_sample, tokenizer, 64, seed=seed)
    val_eval = make_loader(val_ds, tokenizer, 64, seed=seed)
    model = make_model(name, args, device)
    load_report = {"source": "roberta-base_pretrained_no_stsb_checkpoint"}
    init_check = (check_flag_zero(model, next(iter(train_eval)), device)
                  if name == "flag_zero" else {"applicable": False})
    opt = build_e2e_adam(model, model.encoder, args.encoder_lr, args.new_lr)

    # Include epoch 0 so a harmful LTR adaptation cannot silently replace the MSE starting point.
    initial, _ = evaluate(model, val_eval, device)
    train_initial, _ = evaluate(model, train_eval, device)
    best_val, best_epoch = initial["spearman"], 0
    checkpoint = target / "best_model.pt"
    torch.save(model.state_dict(), checkpoint)
    history = [{
        "epoch": 0, "loss": "", "train_mse": "", "train_rank": "",
        "train_sp_sample": train_initial["spearman"], "val_sp": initial["spearman"],
        "val_pe": initial["pearson"], "scale": float(model.log_scale.exp().detach().cpu()),
        "grad_encoder": "", "grad_new": "", "selected_pairs": "", "candidate_pairs": "",
    }]
    write_history(target / "history.csv", history)
    print(f"[start] model={name} variant={variant} seed={seed} "
          f"init_val_sp={best_val:.6f} init_train_sp={train_initial['spearman']:.6f} "
          f"out={target}", flush=True)
    waiting = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        mse_total, rank_total, used, chosen, possible = 0.0, 0.0, 0, 0, 0
        enc_grad_sum, new_grad_sum, grad_steps = 0.0, 0.0, 0
        for t1, t2, labels in loader:
            t1 = {k: v.to(device) for k, v in t1.items()}
            t2 = {k: v.to(device) for k, v in t2.items()}
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            pred, _ = model(t1, t2)
            mse = F.mse_loss(pred, labels / 5.0)
            if variant == "mse_scratch":
                rank = mse * 0.0
                effective, total_pairs = 0, 0
                loss = mse
            elif variant in {"hinge_only", "mse_hinge_all", "mse_hinge_gap"}:
                gap = args.min_label_gap if variant == "mse_hinge_gap" else 0.0
                rank, effective, total_pairs = gap_hinge(pred, labels, gap)
                loss = rank if variant == "hinge_only" else mse + args.rank_weight * rank
            elif variant in {"mse_hinge_margin01", "mse_hinge_labelmargin"}:
                margin_mode = ("fixed01" if variant == "mse_hinge_margin01"
                               else "label_margin")
                rank, effective, total_pairs = calibrated_hinge(pred, labels, margin_mode)
                loss = mse + args.rank_weight * rank
            elif variant == "mse_listnet":
                rank = listnet_kl(pred, labels)
                effective, total_pairs = 0, 0
                loss = mse + args.rank_weight * rank
            else:
                raise ValueError("Unknown variant: " + variant)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at seed={seed}, epoch={epoch}")
            loss.backward()
            enc_norm_sq, new_norm_sq = 0.0, 0.0
            for p in model.encoder.parameters():
                if p.grad is not None:
                    enc_norm_sq += float(p.grad.detach().float().square().sum())
            for n, p in model.named_parameters():
                if not n.startswith("encoder.") and p.grad is not None:
                    new_norm_sq += float(p.grad.detach().float().square().sum())
            enc_grad_sum += math.sqrt(enc_norm_sq)
            new_grad_sum += math.sqrt(new_norm_sq)
            grad_steps += 1
            opt.step()
            count = len(labels)
            mse_total += float(mse.detach()) * count
            rank_total += float(rank.detach()) * count
            used += count
            chosen += effective
            possible += total_pairs

        val_metrics, _ = evaluate(model, val_eval, device)
        tr_metrics, _ = evaluate(model, train_eval, device)
        row = {
            "epoch": epoch, "loss": variant,
            "train_mse": mse_total / used, "train_rank": rank_total / used,
            "train_sp_sample": tr_metrics["spearman"], "val_sp": val_metrics["spearman"],
            "val_pe": val_metrics["pearson"], "scale": float(model.log_scale.exp().detach().cpu()),
            "grad_encoder": enc_grad_sum / grad_steps, "grad_new": new_grad_sum / grad_steps,
            "selected_pairs": chosen, "candidate_pairs": possible,
        }
        history.append(row)
        write_history(target / "history.csv", history)
        if math.isfinite(row["val_sp"]) and row["val_sp"] > best_val:
            best_val, best_epoch, waiting = row["val_sp"], epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            waiting += 1
        print(f"[epoch {epoch:02d}] {name}/{variant}/s{seed} "
              f"mse={row['train_mse']:.6f} rank={row['train_rank']:.6f} "
              f"trSP={row['train_sp_sample']:.6f} valSP={row['val_sp']:.6f} "
              f"best={best_val:.6f}@{best_epoch} scale={row['scale']:.4f} "
              f"pairs={chosen}/{possible}", flush=True)
        if epoch >= args.min_epochs and waiting >= args.patience:
            print(f"[early stop] no val improvement in {waiting} epochs, after min_epochs={args.min_epochs}", flush=True)
            break

    metrics = {
        "model": name, "variant": variant, "seed": seed,
        "protocol": "mean_joint_training_from_pretrained_val_only",
        "initial_val_spearman": initial["spearman"],
        "best_val_spearman": best_val, "best_epoch": best_epoch,
        "epochs_ran": len(history) - 1, "rank_weight": args.rank_weight,
        "min_label_gap": (args.min_label_gap if variant == "mse_hinge_gap" else None),
        "margin_mode": ("fixed01" if variant == "mse_hinge_margin01" else
                        "label_margin" if variant == "mse_hinge_labelmargin" else
                        "fixed1" if variant in {"hinge_only", "mse_hinge_all", "mse_hinge_gap"} else None),
        "encoder_lr": args.encoder_lr, "new_lr": args.new_lr,
        "max_epochs": args.epochs, "min_epochs": args.min_epochs, "patience": args.patience,
        "start": load_report, "equivalence": init_check,
        "test_evaluated": False,
        "notes": "Experimental validation only; no test-set evaluation or test-based selection.",
    }
    (target / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("RESULT " + json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="mean")
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min_epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--rank_weight", type=float, default=0.05)
    parser.add_argument("--min_label_gap", type=float, default=0.5)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--new_lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_root", type=Path, default=Path("outputs/text/stsb_ltr_v2_e20_min12_pat5"))
    parser.add_argument("--output_root", type=Path, default=Path("outputs/text/stsb_ltr_joint_scratch_v1_w005"))
    parser.add_argument("--model_path", default="/home/data/home/wwr_lumos/models/roberta-base")
    parser.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.min_epochs < 1 or args.min_epochs > args.epochs or args.patience < 1 or args.rank_weight < 0 or args.min_label_gap < 0:
        parser.error("Require epochs >= min_epochs >=1, patience >=1, rank_weight >=0")
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        parser.error("At least one nonnegative seed is required")
    if args.output_root.resolve() == args.checkpoint_root.resolve() or args.checkpoint_root.resolve() in args.output_root.resolve().parents:
        parser.error("Output root must be outside source experiment root")
    self_test()
    print(f"source=pretrained_roberta_base model={args.model} variants={args.variants} "
          f"seeds={args.seeds} epochs={args.epochs} min_epochs={args.min_epochs} patience={args.patience} "
          f"rank_weight={args.rank_weight} label_gap={args.min_label_gap}", flush=True)
    if args.dry_run:
        print("[dry_run] checks PASS; no training, no writes", flush=True)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_from_disk(args.stsb_data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    for variant in args.variants:
        for seed in args.seeds:
            train_one(args, raw, tokenizer, seed, variant, device)

    print("\n[FINAL_VALIDATION_COMPARISON] Test is not evaluated.", flush=True)
    for seed in args.seeds:
        baseline_file = args.output_root / "mean" / "mse_scratch" / f"seed_{seed}" / "metrics.json"
        baseline = (json.loads(baseline_file.read_text())["best_val_spearman"]
                    if baseline_file.is_file() else None)
        for variant in args.variants:
            metrics_file = args.output_root / "mean" / variant / f"seed_{seed}" / "metrics.json"
            metrics = json.loads(metrics_file.read_text())
            score = float(metrics["best_val_spearman"])
            delta = f"{score-baseline:+.6f}" if baseline is not None else "NA"
            print(f"seed={seed} variant={variant:<24} best_val_sp={score:.6f} "
                  f"best_epoch={metrics['best_epoch']:02d} vs_mse={delta}", flush=True)


if __name__ == "__main__":
    main()
