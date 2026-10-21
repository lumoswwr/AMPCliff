#!/usr/bin/env python3
"""Read-only audit of STSB LTR training runs (no training, no output mutation).

Examples:
  python text_repro/audit_stsb_ltr_training.py
  python text_repro/audit_stsb_ltr_training.py --root outputs/text/stsb_ltr_ranknet_standard

This diagnostic is intentionally independent of PyTorch and model imports.
It does not rename, rewrite, or delete any saved experiment files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

MODELS = ("mean", "flag_zero", "flag")
LOSSES = (
    "mse", "ranknet", "ranknet_scaled", "pairwise_hinge",
    "listnet", "listmle", "lambdarank",
)
SEEDS = range(5)


def read_history(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def fmt(value):
    return f"{value:.6f}" if math.isfinite(value) else "NA"


def summarize(values):
    good = [v for v in values if math.isfinite(v)]
    return (mean(good), stdev(good) if len(good) > 1 else 0.0) if good else (float("nan"), float("nan"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/text/stsb_ltr_ranknet_standard")
    args = parser.parse_args()
    root = Path(args.root)

    print("STSB LTR TRAINING AUDIT (read-only)")
    print(f"root={root}")
    print("The audit checks training-loss change, validation trajectory, "
          "head scale, and selected checkpoint epoch. "
          "It does not assume that decreasing loss guarantees increasing Spearman.")
    print()

    missing = []
    rows = []
    for model in MODELS:
        for loss in LOSSES:
            for seed in SEEDS:
                d = root / model / loss / f"seed_{seed}"
                p = d / "metrics.json"
                if not p.exists():
                    missing.append(str(p))
                    continue
                with p.open(encoding="utf-8") as handle:
                    m = json.load(handle)
                history = read_history(d / "history.csv")
                if not history:
                    missing.append(str(d / "history.csv"))
                    continue
                first, last = history[0], history[-1]
                val_sp = [numeric(h.get("val_spearman")) for h in history]
                epoch_losses = [numeric(h.get("train_loss")) for h in history]
                scales = [numeric(h.get("score_scale")) for h in history]
                val_change = val_sp[-1] - val_sp[0]
                train_loss_change = epoch_losses[-1] - epoch_losses[0]
                ranknet_pair_counts = [numeric(h.get("effective_pairs")) for h in history]
                rows.append({
                    "model": model,
                    "loss": loss,
                    "seed": seed,
                    "best_epoch": int(m["best_epoch"]),
                    "history_epochs": len(history),
                    "train_sp": numeric(m.get("train_spearman")),
                    "val_sp": numeric(m.get("best_val_spearman")),
                    "test_sp": numeric(m.get("test_spearman")),
                    "epoch1_loss": epoch_losses[0],
                    "last_loss": epoch_losses[-1],
                    "loss_change": train_loss_change,
                    "epoch1_val_sp": val_sp[0],
                    "last_val_sp": val_sp[-1],
                    "val_sp_change": val_change,
                    "first_scale": scales[0],
                    "last_scale": scales[-1],
                    "final_scale": numeric(m.get("final_scale")),
                    "pair_count_first": ranknet_pair_counts[0],
                    "initial_pool_diff": numeric(
                        m.get("initial_equivalence_check", {}).get("max_abs_pool_difference")
                    ),
                })

    if not rows:
        raise SystemExit("No metrics.json + history.csv pairs found under this root.")

    print("PER-CONDITION RESULTS")
    print("model          loss               n  train_sp(avg)  val_sp(avg)  test_sp(avg) "
          "loss_change(avg)  val_change(avg)  last_scale(avg)  best_epoch_counts")
    for model in MODELS:
        for loss in LOSSES:
            sub = [r for r in rows if r["model"] == model and r["loss"] == loss]
            if not sub:
                continue
            means = {k: summarize([r[k] for r in sub])[0] for k in (
                "train_sp", "val_sp", "test_sp", "loss_change",
                "val_sp_change", "last_scale"
            )}
            epochs = dict(sorted(Counter(r["best_epoch"] for r in sub).items()))
            print(
                f'{model:14} {loss:18} {len(sub):1d} '
                f'{fmt(means["train_sp"]):>13} {fmt(means["val_sp"]):>12} '
                f'{fmt(means["test_sp"]):>13} {fmt(means["loss_change"]):>16} '
                f'{fmt(means["val_sp_change"]):>15} {fmt(means["last_scale"]):>16} '
                f'{epochs}'
            )
    print()
    print("INTERPRETATION CHECKS")
    for model in MODELS:
        for loss in LOSSES:
            sub = [r for r in rows if r["model"] == model and r["loss"] == loss]
            if not sub:
                continue
            down = sum(math.isfinite(r["loss_change"]) and r["loss_change"] < 0 for r in sub)
            better_last = sum(math.isfinite(r["val_sp_change"]) and r["val_sp_change"] > 0 for r in sub)
            at_epoch_3 = sum(r["best_epoch"] == 3 for r in sub)
            print(
                f"{model:11} / {loss:16}: "
                f"train loss fell {down}/{len(sub)} seeds; "
                f"last val SP > first val SP {better_last}/{len(sub)}; "
                f"best checkpoint at epoch 3 {at_epoch_3}/{len(sub)}"
            )
    print()
    if missing:
        print(f"MISSING INPUT FILES: {len(missing)}")
        for path in missing[:30]:
            print(f"  {path}")
        if len(missing) > 30:
            print(f"  ... and {len(missing) - 30} more")
        print()
    print("FLAGS TO REVIEW")
    print("* Epoch-3 best checkpoints are evidence that the 3-epoch budget MAY be limiting; not proof.")
    print("* Falling training loss with poor train Spearman indicates objective/metric mismatch or optimization issues, not automatically an implementation bug.")
    print("* Historic raw-cosine RankNet does not train score scale; compare with ranknet_scaled.")
    print("* Current ListMLE arbitrarily orders equal-label STSB examples within a batch; this is a known design defect.")
    print("* Validation/test split differences cannot be used to assert one LTR method is universally superior.")


if __name__ == "__main__":
    main()
