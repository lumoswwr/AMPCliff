#!/usr/bin/env python3
"""MSE vs Hinge-Fixed crossed-loss diagnostics on STSB: 3 poolers x 2 x 5 seeds.

Exactly 20 training epochs, no early stopping. Each training run backpropagates
ONLY its chosen loss, but logs BOTH MSE and Hinge-Fixed on the same minibatches,
before the update, with no gradient on the diagnostic loss.

MSE training matches the established protocol:
    mse(affine_score, labels / 5); trainable log_scale and bias.
Hinge training matches the classical pure ranking protocol:
    hinge_fixed(raw_cosine / 0.05, labels); frozen affine head.

Diagnostic MSE is always mse(model_affine_score, labels/5). The Hinge run's
affine head is fixed at identity, so this is raw-cosine MSE for that run.
raw_cosine_mse is additionally recorded for BOTH regimes as a common-scale
diagnostic. The six-panel plot shows the original, requested MSE objective.

Diagnostic losses do not change training. Validation selects best checkpoint,
but all 20 epochs run even after best validation; test is never accessed.
No pre-existing runs, plots or notes are modified.
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))

MODELS = ("mean", "flag_zero", "flag")
TRAINERS = ("mse", "hinge_fixed")
SEEDS = tuple(range(5))
EPOCHS = 20
TEMPERATURE = 0.05
PROTOCOL = "stsb_mse_hinge_crossloss_fixed20_noearlystop_v1"
DEFAULT_OUTPUT = Path("outputs/text/stsb_mse_hinge_crossloss_fixed20_v1")
PALETTE = {
    "mean": "#0072B2",
    "flag_zero": "#E69F00",
    "flag": "#009E73",
}
MODEL_LABEL = {"mean": "Mean", "flag_zero": "FLaG-Zero", "flag": "FLaG"}
TRAINER_LABEL = {"mse": "MSE", "hinge_fixed": "Hinge Fixed"}


def directory(root: Path, model: str, trainer: str, seed: int) -> Path:
    return root / model / trainer / f"seed_{seed}"


def save_csv(path: Path, rows):
    if not rows:
        raise RuntimeError("No rows to save: " + str(path))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_history(path: Path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate_record(root: Path, model: str, trainer: str, seed: int):
    folder = directory(root, model, trainer, seed)
    mp, hp, cp = folder / "metrics.json", folder / "history.csv", folder / "best_model.pt"
    for target in (mp, hp, cp):
        if not target.is_file():
            raise FileNotFoundError(str(target))
    info = json.loads(mp.read_text(encoding="utf-8"))
    required = {
        "protocol": PROTOCOL, "model": model, "train_objective": trainer,
        "seed": seed, "epochs_ran": EPOCHS, "max_epochs": EPOCHS,
        "temperature": TEMPERATURE, "test_evaluated": False,
        "mse_uses_trainable_affine_head": trainer == "mse",
    }
    for key, expected in required.items():
        if info.get(key) != expected:
            raise RuntimeError(f"Protocol mismatch {mp}: {key}, "
                               f"got={info.get(key)!r} expected={expected!r}")
    if not 0 <= int(info["best_epoch"]) <= EPOCHS:
        raise RuntimeError("Invalid best epoch in " + str(mp))
    rows = load_history(hp)
    if [int(row["epoch"]) for row in rows] != list(range(EPOCHS + 1)):
        raise RuntimeError("Incomplete 0..20 history: " + str(hp))
    for row in rows[1:]:
        for field in ("mse_loss", "hinge_fixed_loss", "raw_cosine_mse",
                      "train_objective_loss", "train_sp_sample", "val_spearman"):
            val = float(row[field])
            if not math.isfinite(val):
                raise RuntimeError(f"Invalid {field} in {hp} epoch {row['epoch']}")
        own = "mse_loss" if trainer == "mse" else "hinge_fixed_loss"
        if abs(float(row["train_objective_loss"]) - float(row[own])) > 1e-8:
            raise RuntimeError("Own objective and logged diagnostic disagree: " + str(hp))
    return rows, info


def verify_all(root: Path):
    for model in MODELS:
        for trainer in TRAINERS:
            for seed in SEEDS:
                validate_record(root, model, trainer, seed)
    print("[CROSSLOSS_VERIFIED] 30/30 runs; all 20 epochs; both losses "
          "on every epoch; Test not read", flush=True)


def prepare_dependencies():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_from_disk
    from torch.utils.data import DataLoader, Subset
    from transformers import AutoTokenizer
    from train_author_text_protocol import (
        PairDataset, PairCollator, seed_everything, build_e2e_adam,
    )
    from run_stsb_ltr_v2 import make_model, check_flag_zero, evaluate
    from stsb_classic_ltr_losses import compute_ltr_loss, self_test
    return locals()


def make_loader(dep, dataset, batch_size: int, shuffle=False, seed=0):
    torch = dep["torch"]
    return dep["DataLoader"](
        dataset, batch_size=batch_size, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=dep["PairCollator"](dep["tokenizer"], 128, torch.float32),
        num_workers=0,
    )


def train_one(args, dep, raw, root: Path, model_name: str, trainer: str, seed: int,
              device):
    target = directory(root, model_name, trainer, seed)
    metric_path = target / "metrics.json"
    if metric_path.is_file():
        validate_record(root, model_name, trainer, seed)
        print(f"[skip completed] {model_name}/{trainer}/seed_{seed}", flush=True)
        return
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Incomplete run, refuse overwrite: " + str(target))
    target.mkdir(parents=True, exist_ok=True)

    torch, F, np = dep["torch"], dep["F"], dep["np"]
    dep["seed_everything"](seed)
    train_ds = dep["PairDataset"](raw["train"], "stsb")
    val_ds = dep["PairDataset"](raw["validation"], "stsb")
    indices = np.random.default_rng(seed).choice(
        len(train_ds), size=min(512, len(train_ds)), replace=False
    )
    train_sample = dep["Subset"](train_ds, sorted(int(i) for i in indices))
    train_dl = make_loader(dep, train_ds, 32, shuffle=True, seed=seed)
    train_sample_dl = make_loader(dep, train_sample, 64, shuffle=False)
    val_dl = make_loader(dep, val_ds, 64, shuffle=False)

    args.model = model_name
    model = dep["make_model"](model_name, args, device)
    equivalence = (
        dep["check_flag_zero"](model, next(iter(train_sample_dl)), device)
        if model_name == "flag_zero" else {"applicable": False}
    )
    # Ranker is trained only on raw cosine. The MSE protocol uses an affine
    # score with trainable scale and bias, as in the previous STSB MSE study.
    if trainer == "hinge_fixed":
        model.log_scale.requires_grad_(False)
        model.bias.requires_grad_(False)
    optimizer = dep["build_e2e_adam"](
        model, model.encoder, args.encoder_lr, args.pool_lr
    )

    initial, _ = dep["evaluate"](model, val_dl, device)
    best_val, best_epoch = float(initial["spearman"]), 0
    checkpoint = target / "best_model.pt"
    torch.save(model.state_dict(), checkpoint)
    rows = [{
        "epoch": 0, "train_objective": trainer,
        "train_objective_loss": "", "mse_loss": "", "hinge_fixed_loss": "",
        "raw_cosine_mse": "", "train_sp_sample": "",
        "val_spearman": best_val, "val_pearson": initial["pearson"],
        "n_train_examples": 0,
    }]
    save_csv(target / "history.csv", rows)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        totals = {"mse_loss": 0.0, "hinge_fixed_loss": 0.0,
                  "raw_cosine_mse": 0.0, "train_objective_loss": 0.0}
        n_seen = 0
        for t1, t2, label in train_dl:
            t1 = {key: val.to(device) for key, val in t1.items()}
            t2 = {key: val.to(device) for key, val in t2.items()}
            label = label.to(device)
            optimizer.zero_grad(set_to_none=True)
            score, cosine = model(t1, t2)
            labels01 = label / 5.0
            if trainer == "mse":
                own_loss = F.mse_loss(score, labels01)
                with torch.no_grad():
                    cross_loss, _, _ = dep["compute_ltr_loss"](
                        "hinge_fixed", cosine.detach(), label,
                        temperature=TEMPERATURE
                    )
                mse_value = own_loss.detach()
                hinge_value = cross_loss.detach()
            else:
                own_loss, _, _ = dep["compute_ltr_loss"](
                    "hinge_fixed", cosine, label, temperature=TEMPERATURE
                )
                with torch.no_grad():
                    cross_loss = F.mse_loss(score.detach(), labels01)
                mse_value = cross_loss.detach()
                hinge_value = own_loss.detach()

            with torch.no_grad():
                raw_mse = F.mse_loss(cosine.detach(), labels01)
            if not all(torch.isfinite(v).item()
                       for v in (own_loss, mse_value, hinge_value, raw_mse)):
                raise FloatingPointError(f"Nonfinite loss {model_name}/{trainer}"
                                         f"/seed_{seed}/epoch_{epoch}")

            # ONLY own_loss drives the optimizer. The cross-loss is detached,
            # and has no gradient or effect on parameter updates.
            own_loss.backward()
            optimizer.step()

            n = int(label.numel())
            totals["train_objective_loss"] += float(own_loss.detach()) * n
            totals["mse_loss"] += float(mse_value) * n
            totals["hinge_fixed_loss"] += float(hinge_value) * n
            totals["raw_cosine_mse"] += float(raw_mse) * n
            n_seen += n

        val_stats, _ = dep["evaluate"](model, val_dl, device)
        train_stats, _ = dep["evaluate"](model, train_sample_dl, device)
        rec = {
            "epoch": epoch, "train_objective": trainer,
            **{key: val / n_seen for key, val in totals.items()},
            "train_sp_sample": float(train_stats["spearman"]),
            "val_spearman": float(val_stats["spearman"]),
            "val_pearson": float(val_stats["pearson"]),
            "n_train_examples": n_seen,
        }
        rows.append(rec)
        save_csv(target / "history.csv", rows)
        if math.isfinite(rec["val_spearman"]) and rec["val_spearman"] > best_val:
            best_val, best_epoch = rec["val_spearman"], epoch
            torch.save(model.state_dict(), checkpoint)
        print(
            f"[epoch {epoch:02d}/20] {model_name}/{trainer}/s{seed} "
            f"mse={rec['mse_loss']:.6f} hinge={rec['hinge_fixed_loss']:.6f} "
            f"valSP={rec['val_spearman']:.6f} best={best_val:.6f}@{best_epoch}",
            flush=True,
        )

    info = {
        "protocol": PROTOCOL, "model": model_name, "train_objective": trainer,
        "seed": seed, "epochs_ran": EPOCHS, "max_epochs": EPOCHS,
        "no_early_stopping": True, "temperature": TEMPERATURE,
        "encoder_lr": args.encoder_lr, "pool_lr": args.pool_lr,
        "mse_uses_trainable_affine_head": trainer == "mse",
        "best_val_spearman": best_val, "best_epoch": best_epoch,
        "test_evaluated": False, "initial_val_spearman": initial["spearman"],
        "equivalence": equivalence,
        "diagnostic_mse": "mse(exp(log_scale)*cosine+bias, label/5)",
        "diagnostic_hinge": "mean(relu(1-sign(delta_y)*delta_cosine/0.05)), ties omitted",
        "mse_native_scaling_note": (
            "MSE-training learns affine score; hinge-training freezes affine "
            "head at identity. A raw-cosine-MSE control is separately logged."
        ),
        "loss_record_timing": "train minibatches, pre-optimizer step",
        "checkpoint_selection": "best validation Spearman, while training all 20 epochs",
    }
    metric_path.write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    validate_record(root, model_name, trainer, seed)
    print(f"[complete] {model_name}/{trainer}/s{seed} best={best_val:.6f}"
          f"@{best_epoch}", flush=True)
    del model, optimizer, train_dl, train_sample_dl, val_dl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect(root):
    all_rows = []
    aggregated = {}
    for model in MODELS:
        for trainer in TRAINERS:
            by_seed = {}
            for seed in SEEDS:
                history, _ = validate_record(root, model, trainer, seed)
                by_seed[seed] = {
                    int(r["epoch"]): r for r in history if int(r["epoch"]) >= 1
                }
                for epoch, r in by_seed[seed].items():
                    all_rows.append({
                        "model": model, "train_method": trainer, "seed": seed,
                        "epoch": epoch, "eval_mse": float(r["mse_loss"]),
                        "eval_hinge_fixed": float(r["hinge_fixed_loss"]),
                        "raw_cosine_mse": float(r["raw_cosine_mse"]),
                        "train_sp_sample": float(r["train_sp_sample"]),
                        "val_spearman": float(r["val_spearman"]),
                    })
            for epoch in range(1, EPOCHS + 1):
                aggregated[(model, trainer, epoch)] = {
                    key: (
                        statistics.mean(float(by_seed[s][epoch][source]) for s in SEEDS),
                        statistics.stdev(float(by_seed[s][epoch][source]) for s in SEEDS),
                    ) for key, source in (
                        ("mse", "mse_loss"), ("hinge_fixed", "hinge_fixed_loss"),
                        ("raw_cosine_mse", "raw_cosine_mse"),
                        ("train_sp_sample", "train_sp_sample"),
                        ("val_spearman", "val_spearman"),
                    )
                }
    if len(all_rows) != 3 * 2 * 5 * EPOCHS:
        raise RuntimeError("Wrong number of per-seed measurements")
    summary_rows = []
    for model in MODELS:
        for trainer in TRAINERS:
            for epoch in range(1, EPOCHS + 1):
                group = aggregated[(model, trainer, epoch)]
                item = {"model": model, "train_method": trainer,
                        "epoch": epoch, "n_seeds": 5}
                for k, (mean, sd) in group.items():
                    item[k + "_mean"] = mean
                    item[k + "_sd"] = sd
                summary_rows.append(item)
    save_csv(root / "crossloss_per_seed_epoch.csv", all_rows)
    save_csv(root / "crossloss_5seed_mean_sd.csv", summary_rows)
    return aggregated


def render(root, aggregated):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    dest = root / "plots"
    dest.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    specs = (
        ("mse", "mse", (0, 0), "Train: MSE | Evaluate: MSE"),
        ("mse", "hinge_fixed", (0, 1), "Train: MSE | Evaluate: Hinge Fixed"),
        ("both", "mse", (0, 2), "Evaluate: MSE | Six Curves"),
        ("hinge_fixed", "mse", (1, 0), "Train: Hinge Fixed | Evaluate: MSE"),
        ("hinge_fixed", "hinge_fixed", (1, 1),
         "Train: Hinge Fixed | Evaluate: Hinge Fixed"),
        ("both", "hinge_fixed", (1, 2),
         "Evaluate: Hinge Fixed | Six Curves"),
    )
    xs = list(range(1, EPOCHS + 1))
    for which_train, metric, loc, title in specs:
        ax = axs[loc]
        selections = TRAINERS if which_train == "both" else (which_train,)
        for trainer in selections:
            for model in MODELS:
                values = [aggregated[(model, trainer, ep)][metric]
                          for ep in xs]
                means = [v[0] for v in values]
                sds = [v[1] for v in values]
                style = "-" if trainer == "mse" else "--"
                color = PALETTE[model]
                opacity = 0.95 if which_train != "both" else 0.90
                ax.plot(xs, means, color=color, linestyle=style,
                        linewidth=2, alpha=opacity,
                        label=f"{MODEL_LABEL[model]} | train {TRAINER_LABEL[trainer]}")
                ax.fill_between(
                    xs,
                    [m - sd for m, sd in zip(means, sds)],
                    [m + sd for m, sd in zip(means, sds)],
                    color=color, alpha=0.10 if which_train != "both" else 0.055,
                )
        ax.set_title(title, fontsize=12, pad=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss" if metric == "mse" else "Hinge Fixed Loss")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlim(1, 20)
        ax.grid(alpha=.24)

    legend_colors = [
        Line2D([0], [0], color=PALETTE[m], lw=2.4,
               label=MODEL_LABEL[m]) for m in MODELS
    ]
    legend_styles = [
        Line2D([0], [0], color="#555555", lw=2.4, linestyle="-",
               label="Train: MSE"),
        Line2D([0], [0], color="#555555", lw=2.4, linestyle="--",
               label="Train: Hinge Fixed"),
    ]
    fig.suptitle("STSB Cross-Loss Evaluation | 20 Epochs | 5 Seeds",
                 fontsize=17, y=.997)
    fig.legend(handles=legend_colors + legend_styles,
               loc="upper center", bbox_to_anchor=(.5, .96),
               ncol=5, frameon=False, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .895])
    figpath = dest / "mse_hinge_crossloss_six_panels.png"
    fig.savefig(figpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {figpath}", flush=True)

    # A second compact view is useful if all six curves overlap in panel 3/6.
    # The requested primary output remains the six-panel figure above.
    print("[CROSSLOSS_DONE] 30 training runs; 6 panels; "
          "all traces contain the same 5 seeds on epochs 1..20", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--model_path", default="/home/data/home/wwr_lumos/models/roberta-base")
    p.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--pool_lr", type=float, default=1e-3)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--verify_only", action="store_true")
    p.add_argument("--plot_only", action="store_true")
    args = p.parse_args()
    if args.encoder_lr <= 0 or args.pool_lr <= 0:
        p.error("Learning rates must be positive")
    if sum([args.dry_run, args.verify_only, args.plot_only]) > 1:
        p.error("Choose only one mode")
    root = args.output_root
    if not root.is_absolute():
        root = REPO / root
    root = root.resolve()
    forbidden = (
        "stsb_soft_spearman_full_3models_t005_w005",
        "stsb_pure_softsp_cosine_v1",
        "stsb_classic_ltr_pure_cosine_v1",
        "stsb_fixed20_no_earlystop_v1",
        "stsb_all_models_ltr_plots_v1",
        "stsb_classic_ltr_final_test_120_frozen_v1",
    )
    if root == REPO or root == (REPO / "outputs").resolve() or (
        any(x in root.parts for x in forbidden)
    ):
        p.error("Output must be a new, independent experiment directory")
    return args, root


def main():
    args, root = parse_args()
    if args.dry_run:
        dep = prepare_dependencies()
        dep["self_test"]()
        torch, F = dep["torch"], dep["F"]
        cosine = torch.tensor([0.3, 0.7, 0.5], requires_grad=True)
        label = torch.tensor([1.0, 5.0, 3.0])
        mse = F.mse_loss(cosine, label / 5)
        hinge, eligible, _ = dep["compute_ltr_loss"](
            "hinge_fixed", cosine, label, temperature=TEMPERATURE
        )
        if eligible != 3 or not (torch.isfinite(mse) and torch.isfinite(hinge)):
            raise RuntimeError("Failed two-objective smoke test")
        mse.backward()
        if cosine.grad is None or not torch.isfinite(cosine.grad).all():
            raise RuntimeError("Failed own-MSE gradient smoke test")
        print("[DRY_RUN_PASS] Pairwise Hinge and MSE tested; "
              "20 epochs, 3 poolers x 2 trainers x 5 seeds = 30 runs. "
              "No training started.", flush=True)
        return

    if args.verify_only or args.plot_only:
        verify_all(root)
        if args.plot_only:
            render(root, collect(root))
        return

    dep = prepare_dependencies()
    dep["self_test"]()
    raw = dep["load_from_disk"](args.stsb_data_path)
    dep["tokenizer"] = dep["AutoTokenizer"].from_pretrained(
        args.model_path, local_files_only=True
    )
    torch = dep["torch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SETUP] device={device} output_root={root} "
          "MSE/hinge-fixed train independently; both losses logged "
          "on identical batches; no early stop; Test untouched", flush=True)
    for model in MODELS:
        for method in TRAINERS:
            for seed in SEEDS:
                train_one(args, dep, raw, root, model, method, seed, device)
    verify_all(root)
    render(root, collect(root))


if __name__ == "__main__":
    main()
