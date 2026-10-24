#!/usr/bin/env python3
"""Corrected classical STSB pairwise/listwise LTR benchmark, validation-only.

Run from pretrained RoBERTa: 3 poolers x 6 pure ranking objectives x 5 seeds.
Use the same model, seed, batches, optimizer, epochs, checkpoint selection and
raw cosine scores as the completed pure Soft Spearman study. Freeze affine
scale/bias. No MSE backprop, no Test reads. Independent output directory.
At completion, compare against existing Mean/FLaG MSE and pure Soft Spearman.
"""
from __future__ import annotations

import argparse
import csv
import gc
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
    PairDataset, PairCollator, seed_everything, build_e2e_adam,
)
from run_stsb_ltr_v2 import (
    make_model, check_flag_zero, evaluate, fit_positive_affine,
)
from stsb_classic_ltr_losses import METHODS, compute_ltr_loss, self_test

MODELS = ("mean", "flag_zero", "flag")
SEEDS = tuple(range(5))
PROTOCOL = "stsb_classic_ltr_fixed_temperature_v1_val_only"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_loader(dataset, tokenizer, batch_size, *, shuffle=False, seed=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=PairCollator(tokenizer, 128, torch.float32),
        num_workers=0,
    )


def run_one(args, raw, tokenizer, model_name, method, seed, device):
    folder = args.output_root / model_name / method / f"seed_{seed}"
    metrics_path = folder / "metrics.json"
    expected = {
        "protocol": PROTOCOL, "model": model_name, "method": method, "seed": seed,
        "max_epochs": args.max_epochs, "min_epochs": args.min_epochs,
        "patience": args.patience, "temperature": args.temperature,
        "label_temperature": args.label_temperature,
        "encoder_lr": args.encoder_lr, "pool_lr": args.pool_lr,
    }
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for field, value in expected.items():
            if metrics.get(field) != value:
                raise ValueError(
                    f"Existing completed run uses different {field}: {folder}, "
                    f"actual={metrics.get(field)!r} expected={value!r}"
                )
        if not (folder / "best_model.pt").is_file() or not (folder / "history.csv").is_file():
            raise RuntimeError(f"Completed record lacks checkpoint/history: {folder}")
        print(f"[skip completed] {model_name}/{method}/seed_{seed}", flush=True)
        return
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Incomplete nonempty output; will not overwrite: {folder}")
    folder.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    train_dataset = PairDataset(raw["train"], "stsb")
    val_dataset = PairDataset(raw["validation"], "stsb")
    sample_indices = np.random.default_rng(seed).choice(
        len(train_dataset), size=min(512, len(train_dataset)), replace=False
    )
    fixed_train_sample = Subset(train_dataset, sorted(int(i) for i in sample_indices))
    train_dl = make_loader(train_dataset, tokenizer, 32, shuffle=True, seed=seed)
    train_eval_dl = make_loader(fixed_train_sample, tokenizer, 64)
    train_full_dl = make_loader(train_dataset, tokenizer, 64)
    val_dl = make_loader(val_dataset, tokenizer, 64)

    args.model = model_name
    model = make_model(model_name, args, device)
    equivalence = (
        check_flag_zero(model, next(iter(train_eval_dl)), device)
        if model_name == "flag_zero" else {"applicable": False}
    )
    # Spearman does not depend on positive affine head transforms. Keep this
    # nuisance head at scale=1, bias=0; only train encoder/pooling via cosine.
    model.log_scale.requires_grad_(False)
    model.bias.requires_grad_(False)
    opt = build_e2e_adam(model, model.encoder, args.encoder_lr, args.pool_lr)

    initial, _ = evaluate(model, val_dl, device)
    best_sp, best_epoch, waiting = initial["spearman"], 0, 0
    checkpoint = folder / "best_model.pt"
    torch.save(model.state_dict(), checkpoint)
    history = [{
        "epoch": 0, "train_rank_loss": "", "train_sp_sample": "",
        "val_spearman": best_sp, "val_pearson": initial["pearson"],
        "train_raw_cos_mse": "", "score_scale": 1.0,
        "eligible_pairs": "", "candidate_pairs": "",
    }]
    write_csv(folder / "history.csv", history)
    print(f"\n[start] {model_name}/{method}/seed_{seed} "
          f"initial_val_sp={best_sp:.6f}, PURE LTR, no MSE backprop", flush=True)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        rank_total, mse_log_total, seen = 0.0, 0.0, 0
        eligible_sum, possible_sum = 0, 0
        for t1, t2, label in train_dl:
            t1 = {key: value.to(device) for key, value in t1.items()}
            t2 = {key: value.to(device) for key, value in t2.items()}
            label = label.to(device)
            opt.zero_grad(set_to_none=True)
            score, cosine = model(t1, t2)
            rank_loss, eligible, possible = compute_ltr_loss(
                method, cosine, label, args.temperature, args.label_temperature
            )
            if not torch.isfinite(rank_loss).item():
                raise FloatingPointError(
                    f"Nonfinite pure-rank loss: {model_name} seed={seed} epoch={epoch}"
                )
            # ONLY pure ranking loss is used for the backward/update.
            rank_loss.backward()
            if epoch == 1 and seen == 0:
                def has_finite_nonzero_gradient(params):
                    present = [p.grad for p in params if p.requires_grad and p.grad is not None]
                    if any(not torch.isfinite(g).all().item() for g in present):
                        raise FloatingPointError("Nonfinite model parameter gradient")
                    return any(g.detach().abs().max().item() > 0.0 for g in present)
                encoder_ok = has_finite_nonzero_gradient(model.encoder.parameters())
                pool_ok = has_finite_nonzero_gradient(model.pool.parameters())
                if not encoder_ok or (
                    any(p.requires_grad for p in model.pool.parameters()) and not pool_ok
                ):
                    raise RuntimeError(
                        f"First-batch gradient disconnected for {model_name}/{method}/seed_{seed}: "
                        f"encoder_ok={encoder_ok} pool_ok={pool_ok}"
                    )
                print(f"[gradient-sanity] {model_name}/{method}/seed_{seed} "
                      f"encoder_has_grad={encoder_ok} pool_has_grad={pool_ok}", flush=True)
            opt.step()
            eligible_sum += eligible
            possible_sum += possible
            n = int(label.numel())
            rank_total += float(rank_loss.detach()) * n
            # Raw cosine MSE is diagnostic only. Never backward through this.
            mse_log_total += float(
                F.mse_loss(cosine.detach(), label / 5.0)
            ) * n
            seen += n

        validation, _ = evaluate(model, val_dl, device)
        sampled_train, _ = evaluate(model, train_eval_dl, device)
        line = {
            "epoch": epoch, "train_rank_loss": rank_total / max(seen, 1),
            "train_sp_sample": sampled_train["spearman"],
            "val_spearman": validation["spearman"],
            "val_pearson": validation["pearson"],
            "train_raw_cos_mse": mse_log_total / max(seen, 1),
            "score_scale": 1.0,
            "eligible_pairs": eligible_sum, "candidate_pairs": possible_sum,
        }
        history.append(line)
        write_csv(folder / "history.csv", history)
        val_sp = line["val_spearman"]
        if math.isfinite(val_sp) and val_sp > best_sp:
            best_sp, best_epoch, waiting = val_sp, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            waiting += 1
        print(
            f"epoch={epoch:02d} model={model_name} method={method} seed={seed} "
            f"pure_rank_loss={line['train_rank_loss']:.6f} "
            f"train_sp_sample={line['train_sp_sample']:.6f} "
            f"val_sp={val_sp:.6f} best={best_sp:.6f}@{best_epoch} "
            f"raw_cos_mse(log_only)={line['train_raw_cos_mse']:.6f} "
            f"eligible_pairs={eligible_sum}/{possible_sum}",
            flush=True,
        )
        if epoch >= args.min_epochs and waiting >= args.patience:
            print(f"[early stop] epoch={epoch}, best_epoch={best_epoch}", flush=True)
            break

    # Selected on validation ONLY. Postselection: train statistics, never test.
    try:
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved, strict=True)
    train_metrics, train_arrays = evaluate(model, train_full_dl, device)
    val_metrics, _ = evaluate(model, val_dl, device)
    scale, intercept = fit_positive_affine(train_arrays["cosine"], train_arrays["labels"])
    calib_train_mse = float(np.mean(
        (scale * train_arrays["cosine"] + intercept -
         train_arrays["labels"] / 5.0) ** 2
    ))

    results = {
        **expected,
        "loss": method, "score_source": "raw_cosine",
        "mse_used_for_backward": False,
        "affine_head_trainable": False,
        "initial_val_spearman": initial["spearman"],
        "best_val_spearman": val_metrics["spearman"],
        "best_epoch": best_epoch,
        "epochs_ran": len(history) - 1,
        "best_train_spearman": train_metrics["spearman"],
        "train_calibrated_mse": calib_train_mse,
        "train_calibration_scale": scale,
        "train_calibration_bias": intercept,
        "mean_equivalence": equivalence,
        "test_evaluated": False,
        "checkpoint_selection": "official validation Spearman only",
        "score_note": "all ranking objectives on raw cosine with fixed temperature; affine head frozen; train MSE calibrated only for diagnostics",
    }
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print("RESULT " + json.dumps(results, ensure_ascii=False), flush=True)
    del model, opt, train_dl, train_eval_dl, train_full_dl, val_dl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def final_summary(args):
    """All-model all-seed tables, with read-only external MSE/SoftSP controls."""
    all_rows = []
    methods = ("mse", "pure_softsp_cosine") + tuple(METHODS)
    sources = {
        "mse": args.reference_root,
        "pure_softsp_cosine": args.softsp_root,
    }
    for model in MODELS:
        for method in methods:
            for seed in SEEDS:
                if method == "mse":
                    p = sources[method] / model / "mse_scratch" / f"seed_{seed}" / "metrics.json"
                    expected_protocol = "softsp_3models_v1_val_only"
                elif method == "pure_softsp_cosine":
                    p = sources[method] / model / "pure_softsp_cosine" / f"seed_{seed}" / "metrics.json"
                    expected_protocol = "stsb_pure_softspearman_cosine_v1_val_only"
                else:
                    p = args.output_root / model / method / f"seed_{seed}" / "metrics.json"
                    expected_protocol = PROTOCOL
                if not p.is_file():
                    print(f"[partial] missing {p}. Full comparison deferred.", flush=True)
                    return
                data = json.loads(p.read_text(encoding="utf-8"))
                if data.get("protocol") != expected_protocol or data.get("model") != model or data.get("seed") != seed:
                    raise RuntimeError(f"Incompatible protocol/model/seed in {p}")
                if method in METHODS and data.get("method") != method:
                    raise RuntimeError(f"Incorrect method recorded in {p}")
                if method == "mse" and data.get("variant") != "mse_scratch":
                    raise RuntimeError(f"Incorrect MSE reference in {p}")
                if method == "pure_softsp_cosine" and data.get("loss") != "pure_soft_midrank_spearman_on_cosine":
                    raise RuntimeError(f"Incorrect SoftSP reference in {p}")
                all_rows.append({
                    "model": model, "method": method, "seed": seed,
                    "val_spearman": float(data["best_val_spearman"]),
                    "best_epoch": int(data["best_epoch"]),
                    "train_spearman": float(
                        data.get("best_train_spearman", data.get("train_spearman", float("nan")))
                    ),
                    "train_calibrated_mse": float(data.get("train_calibrated_mse", float("nan"))),
                    "source": str(p),
                })

    score = {(r["model"], r["method"], r["seed"]): r["val_spearman"] for r in all_rows}
    groups = []
    for model in MODELS:
        for method in methods:
            scores = [score[(model, method, s)] for s in SEEDS]
            group = {
                "model": model, "method": method,
                "category": (
                    "MSE" if method == "mse" else
                    "Listwise" if method in ("pure_softsp_cosine", "listnet_kl", "grouped_listmle") else
                    "Pairwise"
                ),
                "n_seeds": len(SEEDS),
                "val_sp_mean": statistics.mean(scores),
                "val_sp_sd": statistics.stdev(scores),
                "per_seed": json.dumps(scores),
            }
            groups.append(group)

    pairs = []
    for model in MODELS:
        for method in methods:
            if method == "mse":
                continue
            for name, baseline in (
                ("vs_own_mse", (model, "mse")),
                ("vs_mean_same_loss", ("mean", method)),
                ("vs_mean_mse", ("mean", "mse")),
            ):
                deltas = [
                    score[(model, method, s)] - score[(baseline[0], baseline[1], s)]
                    for s in SEEDS
                ]
                pairs.append({
                    "model": model, "method": method, "contrast": name,
                    "mean_delta": statistics.mean(deltas),
                    "sd_delta": statistics.stdev(deltas),
                    "positive_seeds": sum(d > 0 for d in deltas),
                    "per_seed_deltas": json.dumps(deltas),
                })

    write_csv(args.output_root / "validation_all_methods_per_seed.csv", all_rows)
    write_csv(args.output_root / "validation_all_methods_summary.csv", groups)
    write_csv(args.output_root / "validation_all_methods_paired.csv", pairs)
    print("\n[CLASSIC_LTR_FULL_VALIDATION] 120/120 groups x seeds, Test not read.", flush=True)
    print("Model        method                 type       mean      sd         vs_own_MSE")
    for group in groups:
        compare = next(
            (row for row in pairs if row["model"] == group["model"]
             and row["method"] == group["method"] and row["contrast"] == "vs_own_mse"),
            None,
        )
        delta = compare["mean_delta"] if compare else 0.0
        print(f"{group['model']:12} {group['method']:22} {group['category']:10} "
              f"{group['val_sp_mean']:.6f} {group['val_sp_sd']:.6f} {delta:+.6f}", flush=True)
    print("\n[CLASSIC_LTR_PAIRED_VS_MSE]", flush=True)
    for row in pairs:
        if row["contrast"] == "vs_own_mse":
            print(f"{row['model']:10} {row['method']:22} "
                  f"delta={row['mean_delta']:+.6f} sd={row['sd_delta']:.6f} "
                  f"positive={row['positive_seeds']}/5", flush=True)
    print("\n[CLASSIC_LTR_ARCHITECTURE]", flush=True)
    for row in pairs:
        if row["contrast"] == "vs_mean_same_loss" and row["model"] != "mean":
            print(f"{row['method']:22} {row['model']:10} minus_Mean_same_loss="
                  f"{row['mean_delta']:+.6f} sd={row['sd_delta']:.6f} "
                  f"positive_seeds={row['positive_seeds']}/5", flush=True)




def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    p.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--min_epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--label_temperature", type=float, default=0.20)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--pool_lr", type=float, default=1e-3)
    p.add_argument("--output_root", type=Path,
                   default=Path("outputs/text/stsb_classic_ltr_pure_cosine_v1"))
    p.add_argument("--reference_root", type=Path,
                   default=Path("outputs/text/stsb_soft_spearman_full_3models_t005_w005"))
    p.add_argument("--softsp_root", type=Path,
                   default=Path("outputs/text/stsb_pure_softsp_cosine_v1"))
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--model_path",
                   default="/home/data/home/wwr_lumos/models/roberta-base")
    p.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if not args.models or not args.seeds or any(x not in SEEDS for x in args.seeds):
        p.error("Require at least one model and valid seed IDs 0..4")
    if args.min_epochs < 1 or args.max_epochs < args.min_epochs or args.patience < 1:
        p.error("Require max_epochs >= min_epochs >=1 and patience>=1")
    if args.temperature <= 0 or args.label_temperature <= 0 or args.encoder_lr <= 0 or args.pool_lr <= 0:
        p.error("Positive temperature and learning rates required")
    out = args.output_root.resolve()
    for source_dir in (args.reference_root, args.softsp_root):
        source = source_dir.resolve()
        if out == source or out in source.parents or source in out.parents:
            p.error("New pure-LTR outputs must be separate from prior runs")
    self_test()
    print(
        f"[setup] CLASSIC_PAIRWISE_LISTWISE_PURE_COSINE (NO MSE BACKPROP) "
        f"models={args.models} methods={args.methods} seeds={args.seeds} "
        f"temperature={args.temperature} label_temperature={args.label_temperature} "
        f"epochs={args.max_epochs} min_epochs={args.min_epochs} patience={args.patience}",
        flush=True,
    )
    if args.dry_run:
        print("[dry_run] validated loss and args. No data loading, training, or writes.",
              flush=True)
        return

    if args.summary_only:
        final_summary(args)
        return
    raw = load_from_disk(args.stsb_data_path)
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in args.models:
        for method in args.methods:
            for seed in args.seeds:
                run_one(args, raw, tok, model, method, seed, device)
    final_summary(args)


if __name__ == "__main__":
    main()
