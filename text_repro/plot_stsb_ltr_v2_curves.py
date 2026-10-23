#!/usr/bin/env python3
"""Read-only training-curve audit for STSB LTR-v2.

Draws unsmoothed epoch-level train loss, validation Spearman, and score scale
for representative model/loss combinations across all five seeds.
It reads only history.csv and metrics.json. Old training outputs are untouched.

Usage:
  python text_repro/plot_stsb_ltr_v2_curves.py
  python text_repro/plot_stsb_ltr_v2_curves.py --pairs mean/mse mean/pairwise_hinge
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PAIRS = (
    "mean/mse",                    # strong pointwise reference
    "mean/pairwise_hinge",         # improved with longer training
    "flag_zero/pairwise_hinge",    # same loss, different architecture
    "flag_zero/ranknet_scaled",    # high train-SP variance
    "flag/listnet",               # high train-SP variance
    "mean/ranknet",               # v2 aggregate exactly matches v1
    "mean/lambdarank",            # v2 aggregate exactly matches v1
    "flag_zero/listmle",          # strongest test LTR for FLaG-Zero
)


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def curve_data(history, metric):
    xs, ys = [], []
    for row in history:
        epoch, value = number(row.get("epoch")), number(row.get(metric))
        if math.isfinite(epoch) and math.isfinite(value):
            xs.append(int(epoch))
            ys.append(value)
    return xs, ys


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="outputs/text/stsb_ltr_v2_e20_min12_pat5",
        help="Folder containing <model>/<loss>/seed_N/history.csv",
    )
    parser.add_argument(
        "--out", default="outputs/text/stsb_ltr_v2_curve_audit",
        help="Separate folder for derived PNGs and curve_audit.csv",
    )
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    args = parser.parse_args()

    source = Path(args.root)
    dest = Path(args.out)
    if source.resolve() == dest.resolve() or source.resolve() in dest.resolve().parents:
        parser.error("--out must be outside the original experiment output tree")
    if any(seed < 0 for seed in args.seeds):
        parser.error("Seeds must be nonnegative")
    dest.mkdir(parents=True, exist_ok=True)

    records = []
    for entry in args.pairs:
        try:
            model, loss = entry.split("/")
        except ValueError:
            parser.error(f"Invalid --pairs entry {entry!r}; expected model/loss")
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
        series = (
            ("train_loss", "Train objective (epoch average)"),
            ("val_spearman", "Validation Spearman"),
            ("score_scale", "Score scale exp(log_scale)"),
        )
        any_data = False
        for seed in args.seeds:
            run = source / model / loss / f"seed_{seed}"
            history = read_rows(run / "history.csv")
            if not history:
                print(f"[missing] {run / 'history.csv'}", flush=True)
                continue
            any_data = True
            metrics_path = run / "metrics.json"
            metrics = {}
            if metrics_path.exists():
                with metrics_path.open(encoding="utf-8") as fh:
                    metrics = json.load(fh)
            best_epoch = metrics.get("best_epoch")
            if best_epoch is None:
                ranked = [
                    (number(row.get("val_spearman")), int(number(row["epoch"])))
                    for row in history
                    if math.isfinite(number(row.get("val_spearman")))
                ]
                best_epoch = max(ranked)[1] if ranked else None

            for ax, (key, label) in zip(axes, series):
                xs, ys = curve_data(history, key)
                if not xs:
                    continue
                line, = ax.plot(xs, ys, marker="o", markersize=2.3, linewidth=1.2, label=f"seed {seed}")
                if key == "val_spearman" and best_epoch is not None and int(best_epoch) in xs:
                    k = xs.index(int(best_epoch))
                    ax.scatter([xs[k]], [ys[k]], s=42, facecolors="none",
                               edgecolors=line.get_color(), linewidths=1.7, zorder=4)
                ax.set_title(label)
                ax.set_xlabel("Epoch")
                ax.grid(alpha=0.24)
                ax.axvline(3, linestyle="--", linewidth=1, color="gray", alpha=0.65)
            losses = [number(row.get("train_loss")) for row in history]
            vals = [number(row.get("val_spearman")) for row in history]
            scales = [number(row.get("score_scale")) for row in history]
            finite_vals = [x for x in vals if math.isfinite(x)]
            first_loss, last_loss = losses[0], losses[-1]
            records.append({
                "model": model,
                "loss": loss,
                "seed": seed,
                "epochs_ran": len(history),
                "best_epoch": best_epoch,
                "best_val_spearman": max(finite_vals) if finite_vals else "",
                "last_val_spearman": vals[-1],
                "train_loss_first": first_loss,
                "train_loss_last": last_loss,
                "train_loss_relative_change": (
                    (last_loss - first_loss) / abs(first_loss)
                    if first_loss != 0 and math.isfinite(first_loss + last_loss) else ""
                ),
                "scale_first": scales[0],
                "scale_last": scales[-1],
                "best_in_first_3": best_epoch is not None and int(best_epoch) <= 3,
                "best_at_max_epochs": (
                    best_epoch is not None and int(best_epoch) == 20
                ),
            })

        if any_data:
            axes[0].legend(fontsize=8)
            fig.suptitle(
                f"{model} / {loss} | all seeds, raw epoch values\n"
                "Dashed vertical line: old 3-epoch budget; open circle: selected best epoch",
                fontsize=11,
            )
            output = dest / f"{model}__{loss}.png"
            fig.savefig(output, dpi=170, bbox_inches="tight")
            print(f"[plot] {output}", flush=True)
        plt.close(fig)

    if records:
        csv_file = dest / "curve_audit.csv"
        write_csv(csv_file, records)
        print(f"\n[audit] {csv_file}\n", flush=True)
        print("CONDITION | seeds | best<=3 | best_at_20 | median_epochs | "
              "loss_decreased | val_peak_vs_last")
        for entry in args.pairs:
            m, loss = entry.split("/")
            rr = [r for r in records if r["model"] == m and r["loss"] == loss]
            if not rr:
                continue
            n = len(rr)
            best_early = sum(bool(r["best_in_first_3"]) for r in rr)
            best_limit = sum(bool(r["best_at_max_epochs"]) for r in rr)
            fell = sum(
                isinstance(r["train_loss_relative_change"], float)
                and r["train_loss_relative_change"] < 0 for r in rr
            )
            epochs = sorted(r["epochs_ran"] for r in rr)
            median = epochs[len(epochs) // 2]
            drops = [
                r["best_val_spearman"] - r["last_val_spearman"]
                for r in rr
                if isinstance(r["best_val_spearman"], float)
                and math.isfinite(r["last_val_spearman"])
            ]
            avg_drop = sum(drops) / len(drops) if drops else float("nan")
            print(f"{entry:29} | {n} | {best_early} | {best_limit} | "
                  f"{median} | {fell}/{n} | {avg_drop:.6f}")

        print("\nNOTES:")
        print("- Epoch-average loss is NOT minibatch-level smoothness.")
        print("- Falling train loss with falling validation Spearman can indicate overfitting.")
        print("- Best epoch <=3 can explain identical selected checkpoints in v1/v2.")
        print("- Raw RankNet cannot learn the affine score scale by design.")
        print("- Training convergence and held-out generalization are separate checks.")
    else:
        raise SystemExit("No history.csv files found. Verify --root and run locations.")


if __name__ == "__main__":
    main()
