#!/usr/bin/env python3
"""Plot comparable 20-epoch STS-B loss/Spearman curves from all 120 runs.

Every plotted epoch (1..20) is the mean of EXACTLY the same 5 seeds; early
stopping is disallowed and verified. Each subplot compares three pooling
architectures under one fixed objective, avoiding cross-loss scale comparisons.

Outputs: 3 2x4-panel PNGs + full per-seed table + 5-seed aggregate table.
"""
from __future__ import print_function

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_stsb_fixed20_no_earlystop import (
    MODELS, METHODS, SEEDS, EPOCHS, source_dir, verify_all,
)

LABELS = {
    "mse": "MSE",
    "pure_softsp_cosine": "Soft Spearman",
    "ranknet": "RankNet",
    "hinge_fixed": "Hinge Fixed",
    "hinge_label": "Hinge Label",
    "lambda_ndcg": "LambdaRank-style",
    "listnet_kl": "ListNet KL",
    "grouped_listmle": "Grouped ListMLE",
}
MODEL_LABELS = {
    "mean": "Mean",
    "flag_zero": "FLaG-Zero",
    "flag": "FLaG",
}
COLORS = {
    "mean": "#0072B2",
    "flag_zero": "#E69F00",
    "flag": "#009E73",
}
FIELDS = (
    ("loss", "Training Loss",
     "STSB: Training Loss, 20 Complete Epochs, Five Seeds",
     "pooling_overlay_train_loss_fixed20.png"),
    ("train", "Sampled Train Spearman",
     "STSB: Sampled Train Spearman, 20 Epochs, Five Seeds",
     "pooling_overlay_train_spearman_fixed20.png"),
    ("validation", "Validation Spearman",
     "STSB: Validation Spearman, 20 Epochs, Five Seeds",
     "pooling_overlay_validation_spearman_fixed20.png"),
)


def read_histories(root):
    data = {}
    for method in METHODS:
        for model in MODELS:
            paths = [source_dir(root, model, method, seed) / "history.csv"
                     for seed in SEEDS]
            rows_by_seed = []
            for path in paths:
                with path.open("r", newline="", encoding="utf-8") as fp:
                    rows = list(csv.DictReader(fp))
                if [int(row["epoch"]) for row in rows] != list(range(EPOCHS + 1)):
                    raise RuntimeError("Not all 20 epochs in %s" % path)
                loss_col = ("train_mse" if method == "mse"
                            else "train_rank_loss")
                val_col = ("val_sp" if method == "mse" else "val_spearman")
                records = {}
                for row in rows[1:]:
                    ep = int(row["epoch"])
                    records[ep] = {
                        "loss": float(row[loss_col]),
                        "train": float(row["train_sp_sample"]),
                        "validation": float(row[val_col]),
                    }
                    if not all(math.isfinite(x) for x in records[ep].values()):
                        raise RuntimeError("Nonfinite log value in %s epoch=%d" %
                                           (path, ep))
                rows_by_seed.append(records)
            data[(model, method)] = rows_by_seed
    return data


def metric_curves(data, model, method, field):
    epochs = list(range(1, EPOCHS + 1))
    values = [
        [data[(model, method)][seed][ep][field] for seed in SEEDS]
        for ep in epochs
    ]
    assert all(len(values_at_epoch) == 5 for values_at_epoch in values)
    means = [statistics.mean(v) for v in values]
    sds = [statistics.stdev(v) for v in values]
    return epochs, means, sds


def render_grid(data, root, field, ylabel, title, filename):
    fig, axs = plt.subplots(2, 4, figsize=(21, 10))
    for k, method in enumerate(METHODS):
        ax = axs.flat[k]
        for model in MODELS:
            xs, means, sds = metric_curves(data, model, method, field)
            lower = [a - b for a, b in zip(means, sds)]
            upper = [a + b for a, b in zip(means, sds)]
            ax.plot(xs, means, linewidth=1.8, color=COLORS[model],
                    label=MODEL_LABELS[model])
            ax.fill_between(xs, lower, upper, color=COLORS[model], alpha=.13)
        ax.set_title(LABELS[method], fontsize=12, pad=7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 20)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=.23)
    handles, legends = axs.flat[0].get_legend_handles_labels()
    fig.suptitle(title, y=.99, fontsize=17)
    fig.legend(handles, legends, loc="upper center", bbox_to_anchor=(.5, .963),
               ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, .915])
    target = root / filename
    fig.savefig(str(target), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[saved] %s" % target, flush=True)


def dump_tables(data, root):
    per_seed = []
    summary = []
    for method in METHODS:
        for model in MODELS:
            bests = []
            lasts = []
            losses1 = []
            losses20 = []
            for seed in SEEDS:
                rows = data[(model, method)][seed]
                path = source_dir(root.parent, model, method, seed)
                import json
                cfg = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
                best = float(cfg["best_val_spearman"])
                last = rows[20]["validation"]
                rec = {
                    "model": model, "method": method, "seed": seed,
                    "best_epoch": int(cfg["best_epoch"]),
                    "best_val_spearman": best,
                    "epoch20_val_spearman": last,
                    "epoch20_train_sp_sample": rows[20]["train"],
                    "epoch1_train_loss": rows[1]["loss"],
                    "epoch20_train_loss": rows[20]["loss"],
                    "epochs_ran": int(cfg["epochs_ran"]),
                }
                per_seed.append(rec)
                bests.append(best)
                lasts.append(last)
                losses1.append(rows[1]["loss"])
                losses20.append(rows[20]["loss"])
            summary.append({
                "model": model, "method": method,
                "n_seeds_all_epochs": 5,
                "best_val_sp_mean": statistics.mean(bests),
                "best_val_sp_sd": statistics.stdev(bests),
                "epoch20_val_sp_mean": statistics.mean(lasts),
                "epoch20_val_sp_sd": statistics.stdev(lasts),
                "mean_loss_epoch1": statistics.mean(losses1),
                "mean_loss_epoch20": statistics.mean(losses20),
                "note": "best checkpoint still selected by validation Spearman",
            })
    for name, rows in (
        ("per_seed_best_vs_epoch20.csv", per_seed),
        ("summary_best_vs_epoch20.csv", summary),
    ):
        path = root / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("[saved] %s" % path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path,
                        default=Path("outputs/text/stsb_fixed20_no_earlystop_v1"))
    args = parser.parse_args()
    root = args.input_root.resolve()
    verify_all(root)
    data = read_histories(root)
    dest = root / "plots"
    dest.mkdir(parents=True, exist_ok=True)
    for field, ylabel, title, filename in FIELDS:
        render_grid(data, dest, field, ylabel, title, filename)
    dump_tables(data, dest)
    print("[PLOTS_COMPLETE] 3 pooling comparisons, 5-seed means at each of "
          "20 epochs; source checkpoints untouched.", flush=True)


if __name__ == "__main__":
    main()
