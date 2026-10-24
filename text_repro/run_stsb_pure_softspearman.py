#!/usr/bin/env python3
"""PURE rank-loss STSB benchmark: Mean, FLaG-Zero, original FLaG.

No MSE is used for backpropagation, and TEST data is never read.
Loss: 1 - Pearson(soft ranks of raw cosine, exact tied label midranks).
Controls (existing 3-model MSE runs) are read for validation comparisons only.
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
from run_stsb_soft_spearman_full import (
    soft_midrank_spearman_loss, self_test,
)

MODELS = ("mean", "flag_zero", "flag")
SEEDS = tuple(range(5))
PROTOCOL = "stsb_pure_softspearman_cosine_v1_val_only"


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


def run_one(args, raw, tokenizer, model_name, seed, device):
    folder = args.output_root / model_name / "pure_softsp_cosine" / f"seed_{seed}"
    metrics_path = folder / "metrics.json"
    expected = {
        "protocol": PROTOCOL, "model": model_name, "seed": seed,
        "max_epochs": args.max_epochs, "min_epochs": args.min_epochs,
        "patience": args.patience, "temperature": args.temperature,
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
        print(f"[skip completed] {model_name}/seed_{seed}", flush=True)
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
    }]
    write_csv(folder / "history.csv", history)
    print(f"\n[start] {model_name}/pure_softsp_cosine/seed_{seed} "
          f"initial_val_sp={best_sp:.6f}, no MSE backprop", flush=True)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        rank_total, mse_log_total, seen = 0.0, 0.0, 0
        for t1, t2, label in train_dl:
            t1 = {key: value.to(device) for key, value in t1.items()}
            t2 = {key: value.to(device) for key, value in t2.items()}
            label = label.to(device)
            opt.zero_grad(set_to_none=True)
            score, cosine = model(t1, t2)
            rank_loss = soft_midrank_spearman_loss(cosine, label, args.temperature)
            if not torch.isfinite(rank_loss).item():
                raise FloatingPointError(
                    f"Nonfinite pure-rank loss: {model_name} seed={seed} epoch={epoch}"
                )
            # ONLY ranking loss is used for the backward/update.
            rank_loss.backward()
            opt.step()
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
            f"epoch={epoch:02d} model={model_name} seed={seed} "
            f"pure_rank_loss={line['train_rank_loss']:.6f} "
            f"train_sp_sample={line['train_sp_sample']:.6f} "
            f"val_sp={val_sp:.6f} best={best_sp:.6f}@{best_epoch} "
            f"raw_cos_mse(log_only)={line['train_raw_cos_mse']:.6f}",
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
        "loss": "pure_soft_midrank_spearman_on_cosine",
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
        "score_note": "affine head frozen; train MSE calibrated using train labels for diagnostics only",
    }
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print("RESULT " + json.dumps(results, ensure_ascii=False), flush=True)
    del model, opt, train_dl, train_eval_dl, train_full_dl, val_dl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def final_summary(args):
    all_scores = {}
    for model in MODELS:
        for seed in SEEDS:
            p = args.output_root / model / "pure_softsp_cosine" / f"seed_{seed}" / "metrics.json"
            if not p.is_file():
                print(f"[partial] missing {p}, no complete summary", flush=True)
                return
            all_scores[(model, seed)] = json.loads(p.read_text(encoding="utf-8"))

    controls = {}
    for model in MODELS:
        for seed in SEEDS:
            p = args.reference_root / model / "mse_scratch" / f"seed_{seed}" / "metrics.json"
            if not p.is_file():
                raise FileNotFoundError(f"Read-only MSE baseline missing: {p}")
            control = json.loads(p.read_text(encoding="utf-8"))
            if (control.get("protocol") != "softsp_3models_v1_val_only"
                    or control.get("model") != model
                    or control.get("seed") != seed
                    or control.get("variant") != "mse_scratch"):
                raise RuntimeError(f"Reference protocol mismatch: {p}")
            controls[(model, seed)] = float(control["best_val_spearman"])

    rows, paired = [], []
    for model in MODELS:
        values = [float(all_scores[(model, seed)]["best_val_spearman"]) for seed in SEEDS]
        rows.append({
            "model": model, "loss": "pure_softsp_cosine", "n": 5,
            "mean_val_sp": statistics.mean(values),
            "sd_val_sp": statistics.stdev(values),
            "per_seed_val": json.dumps(values),
        })
        for name, baselines in (
            ("pure_minus_own_mse", [controls[(model, seed)] for seed in SEEDS]),
            ("pure_minus_mean_mse", [controls[("mean", seed)] for seed in SEEDS]),
        ):
            ds = [x - y for x, y in zip(values, baselines)]
            paired.append({
                "model": model, "comparison": name,
                "mean_delta": statistics.mean(ds),
                "sd_delta": statistics.stdev(ds),
                "positive_seed_count": sum(x > 0 for x in ds),
                "per_seed_deltas": json.dumps(ds),
            })

    write_csv(args.output_root / "pure_summary_val.csv", rows)
    write_csv(args.output_root / "pure_paired_vs_mse_val.csv", paired)
    print("\n[PURE_LTR_VALIDATION_SUMMARY] 15/15, no test evaluated", flush=True)
    for row in rows:
        print(
            f"{row['model']:10} pureSP val={row['mean_val_sp']:.6f} "
            f"±{row['sd_val_sp']:.6f} per_seed={row['per_seed_val']}",
            flush=True,
        )
    print("\n[PURE_LTR_PAIRED_VS_MSE]", flush=True)
    for row in paired:
        print(
            f"{row['model']:10} {row['comparison']:20} "
            f"delta={row['mean_delta']:+.6f} "
            f"sd={row['sd_delta']:.6f} "
            f"positive={row['positive_seed_count']}/5 "
            f"per_seed={row['per_seed_deltas']}",
            flush=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--min_epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--pool_lr", type=float, default=1e-3)
    p.add_argument("--output_root", type=Path,
                   default=Path("outputs/text/stsb_pure_softsp_cosine_v1"))
    p.add_argument("--reference_root", type=Path,
                   default=Path("outputs/text/stsb_soft_spearman_full_3models_t005_w005"))
    p.add_argument("--model_path",
                   default="/home/data/home/wwr_lumos/models/roberta-base")
    p.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if not args.models or not args.seeds or any(x not in SEEDS for x in args.seeds):
        p.error("Require at least one model and valid seed IDs 0..4")
    if args.min_epochs < 1 or args.max_epochs < args.min_epochs or args.patience < 1:
        p.error("Require max_epochs >= min_epochs >=1 and patience>=1")
    if args.temperature <= 0 or args.encoder_lr <= 0 or args.pool_lr <= 0:
        p.error("Positive temperature and learning rates required")
    source = args.reference_root.resolve()
    out = args.output_root.resolve()
    if out == source or out in source.parents or source in out.parents:
        p.error("Pure LTR outputs must be separate from prior MSE controls")
    self_test()
    print(
        f"[setup] PURE_SOFT_SPEARMAN_COSINE (NO MSE BACKPROP) "
        f"models={args.models} seeds={args.seeds} temperature={args.temperature} "
        f"epochs={args.max_epochs} min_epochs={args.min_epochs} patience={args.patience}",
        flush=True,
    )
    if args.dry_run:
        print("[dry_run] validated loss and args. No data loading, training, or writes.",
              flush=True)
        return

    raw = load_from_disk(args.stsb_data_path)
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in args.models:
        for seed in args.seeds:
            run_one(args, raw, tok, model, seed, device)
    final_summary(args)


if __name__ == "__main__":
    main()
