#!/usr/bin/env python3
"""Formal five-seed STSB Soft Spearman validation-only comparison.

Models: Mean, mean-equivalent FLaG-Zero and original FLaG.
Each model: matched MSE and MSE+0.05 Soft Spearman objective.
Retains the verified minibatch tie-aware soft-rank loss from the seed-0 probe.
Selections ONLY by official validation Spearman; test split is not read.
Creates isolated results with read-only skip of completed, matching runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
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

VARIANTS = ("mse_scratch", "mse_softsp_w005")
MODELS = ("mean", "flag_zero", "flag")


def soft_midrank_spearman_loss(scores, labels, temperature=0.05):
    """1 - Pearson(soft ranks of scores, exact average ranks of labels).

    Soft ascending rank_i = 0.5 + sum_j sigmoid((score_i-score_j)/T).
    This includes self-comparison sigmoid(0)=0.5.
    Ground-truth midranks use exact tie handling, matching scipy's
    Spearman rankdata(method='average') semantics within each minibatch.

    This is a differentiable *minibatch* proxy, not exact full-data Spearman.
    """
    if scores.numel() < 3:
        return scores.sum() * 0.0
    with torch.no_grad():
        label_difference = labels[:, None] - labels[None, :]
        true_ranks = (
            0.5
            + (label_difference > 0).to(scores.dtype).sum(dim=1)
            + 0.5 * (label_difference == 0).to(scores.dtype).sum(dim=1)
        )
        true_centered = true_ranks - true_ranks.mean()
        true_sq = true_centered.square().sum()
    if float(true_sq) <= 1e-12:
        return scores.sum() * 0.0
    scaled_diffs = (scores[:, None] - scores[None, :]) / temperature
    soft_ranks = 0.5 + torch.sigmoid(scaled_diffs).sum(dim=1)
    pred_centered = soft_ranks - soft_ranks.mean()
    pred_sq = pred_centered.square().sum()
    rho = (pred_centered * true_centered).sum() / (
        torch.sqrt(pred_sq + 1e-8) * torch.sqrt(true_sq + 1e-8)
    )
    return 1.0 - rho


def self_test():
    labels = torch.tensor([0.0, 0.0, 1.0, 3.0, 5.0])
    good = torch.tensor([0.1, 0.1, 0.3, 0.7, 0.9], requires_grad=True)
    reversed_scores = torch.tensor([0.9, 0.9, 0.7, 0.3, 0.1])
    good_loss = soft_midrank_spearman_loss(good, labels)
    wrong_loss = soft_midrank_spearman_loss(reversed_scores, labels)
    if not (torch.isfinite(good_loss).item() and torch.isfinite(wrong_loss).item()
            and good_loss.item() < wrong_loss.item()):
        raise RuntimeError("Soft-Spearman ranking-direction check failed")
    good_loss.backward()
    if good.grad is None or not torch.isfinite(good.grad).all().item():
        raise RuntimeError("Soft-Spearman gradient self-test failed")
    ties = torch.ones_like(labels)
    loss_ties = soft_midrank_spearman_loss(good, ties)
    if loss_ties.item() != 0.0:
        raise RuntimeError("All-tied labels should contribute zero ranking loss")
    print("[self-test] tie-aware Soft-Spearman objective and gradients PASS", flush=True)


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
    metrics_path = target / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        expected = {
            "protocol": "softsp_3models_v1_val_only",
            "model": name, "variant": variant, "seed": seed,
            "max_epochs": args.epochs, "min_epochs": args.min_epochs,
            "patience": args.patience, "rank_temperature": args.temperature,
            "rank_weight": (0.05 if variant == "mse_softsp_w005" else 0.0),
            "encoder_lr": args.encoder_lr, "new_lr": args.new_lr,
        }
        for key, value in expected.items():
            if metrics.get(key) != value:
                raise RuntimeError(f"Existing run has incompatible {key}: "
                                   f"{target} has {metrics.get(key)!r}, expected {value!r}")
        if not (target / "best_model.pt").is_file() or not (target / "history.csv").is_file():
            raise RuntimeError(f"Completed metrics but missing checkpoint/history: {target}")
        print(f"[skip completed] {name}/{variant}/seed_{seed}", flush=True)
        return
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite incomplete output: {target}; "
                              "investigate partial files or move to a new --output_root")
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
            elif variant == "mse_softsp_w005":
                rank = soft_midrank_spearman_loss(pred, labels, args.temperature)
                effective, total_pairs = 0, 0
                loss = mse + 0.05 * rank
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
        "protocol": "softsp_3models_v1_val_only",
        "initial_val_spearman": initial["spearman"],
        "best_val_spearman": best_val, "best_epoch": best_epoch,
        "epochs_ran": len(history) - 1,
        "rank_weight": 0.0 if variant == "mse_scratch" else 0.05,
        "rank_temperature": args.temperature,
        "encoder_lr": args.encoder_lr, "new_lr": args.new_lr,
        "max_epochs": args.epochs, "min_epochs": args.min_epochs, "patience": args.patience,
        "start": load_report, "equivalence": init_check,
        "test_evaluated": False,
        "notes": "Minibatch tie-aware soft Spearman surrogate; validation-only; not exact global Spearman.",
    }
    (target / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("RESULT " + json.dumps(metrics), flush=True)



def write_summary_if_complete(root, args):
    """Summarize only if all 3x2x5 standard conditions are complete."""
    entries = {}
    for model in MODELS:
        for variant in VARIANTS:
            for seed in range(5):
                path = root / model / variant / f"seed_{seed}" / "metrics.json"
                if not path.is_file():
                    print(f"[partial coverage] missing {path}; summary deferred.", flush=True)
                    return
                entries[(model, variant, seed)] = json.loads(
                    path.read_text(encoding="utf-8")
                )

    def mean_sd(xs):
        return statistics.mean(xs), statistics.stdev(xs)

    grouped, paired = [], []
    for model in MODELS:
        for variant in VARIANTS:
            scores = [
                float(entries[(model, variant, s)]["best_val_spearman"])
                for s in range(5)
            ]
            best_epochs = [
                int(entries[(model, variant, s)]["best_epoch"])
                for s in range(5)
            ]
            avg, std = mean_sd(scores)
            grouped.append({
                "model": model, "variant": variant,
                "n_seeds": len(scores), "val_sp_mean": avg,
                "val_sp_sample_sd": std,
                "best_epoch_mean": statistics.mean(best_epochs),
                "per_seed_val": json.dumps(scores),
                "per_seed_best_epoch": json.dumps(best_epochs),
            })
        relative_own = [
            float(entries[(model, "mse_softsp_w005", s)]["best_val_spearman"]) -
            float(entries[(model, "mse_scratch", s)]["best_val_spearman"])
            for s in range(5)
        ]
        relative_mean = [
            float(entries[(model, "mse_softsp_w005", s)]["best_val_spearman"]) -
            float(entries[("mean", "mse_scratch", s)]["best_val_spearman"])
            for s in range(5)
        ]
        for comparison, deltas in (
            ("softsp_minus_own_mse", relative_own),
            ("softsp_minus_mean_mse", relative_mean),
        ):
            avg, std = mean_sd(deltas)
            paired.append({
                "model": model, "comparison": comparison,
                "mean_delta_val_sp": avg, "sample_sd_delta": std,
                "improved_seed_count": sum(x > 0 for x in deltas),
                "per_seed_deltas": json.dumps(deltas),
            })

    for filename, records in (
        ("summary_val.csv", grouped),
        ("paired_val_differences.csv", paired),
    ):
        path = root / filename
        # Derived summary files can be refreshed after all 30 runs complete;
        # individual training outputs are never overwritten.
        write_history(path, records)

    print("\n[FINAL_VALIDATION_SUMMARY] 30/30 conditions; no test evaluated.", flush=True)
    for row in grouped:
        print(f"{row['model']:10} {row['variant']:18} "
              f"val_sp={row['val_sp_mean']:.6f} ±{row['val_sp_sample_sd']:.6f} "
              f"mean_best_epoch={row['best_epoch_mean']:.1f}", flush=True)
    print("\n[PAIRED_VALIDATION_DELTAS]", flush=True)
    for row in paired:
        print(f"{row['model']:10} {row['comparison']:23} "
              f"mean_delta={row['mean_delta_val_sp']:+.6f} "
              f"std={row['sample_sd_delta']:.6f} "
              f"positive_seeds={row['improved_seed_count']}/5 "
              f"per_seed={row['per_seed_deltas']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min_epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--new_lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_root", type=Path, default=Path("outputs/text/stsb_ltr_v2_e20_min12_pat5"))
    parser.add_argument("--output_root", type=Path, default=Path("outputs/text/stsb_soft_spearman_full_3models_t005_w005"))
    parser.add_argument("--model_path", default="/home/data/home/wwr_lumos/models/roberta-base")
    parser.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.min_epochs < 1 or args.min_epochs > args.epochs or args.patience < 1 or args.temperature <= 0:
        parser.error("Require epochs >= min_epochs >=1, patience >=1, temperature >0")
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        parser.error("At least one nonnegative seed is required")
    if args.output_root.resolve() == args.checkpoint_root.resolve() or args.checkpoint_root.resolve() in args.output_root.resolve().parents:
        parser.error("Output root must be outside source experiment root")
    if args.output_root.resolve() == Path("outputs/text/stsb_soft_spearman_v1_T005").resolve():
        parser.error("Must not write into the existing seed-0 pilot's output root")
    self_test()
    print(f"source=pretrained_roberta_base models={args.models} variants={args.variants} "
          f"seeds={args.seeds} epochs={args.epochs} min_epochs={args.min_epochs} "
          f"patience={args.patience} soft_rank_temperature={args.temperature}", flush=True)
    if args.dry_run:
        print("[dry_run] checks PASS; no training, no writes", flush=True)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_from_disk(args.stsb_data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    for model in args.models:
        args.model = model
        for variant in args.variants:
            for seed in args.seeds:
                train_one(args, raw, tokenizer, seed, variant, device)

    write_summary_if_complete(args.output_root, args)


if __name__ == "__main__":
    main()
