#!/usr/bin/env python3
"""Repeat the 120-condition STS-B ranking comparison for exactly 20 epochs.

No early stopping: min_epochs=20, patience=21 and max_epochs=20 in ALL three
existing training programs.  The stop predicate cannot be true on epochs 1..20.
All original seeds, loss functions, architectures, optimizers and data splits
are unchanged. Best checkpoint is still selected by VALIDATION Spearman.

The output root is deliberately NEW. Source results are never overwritten.
No test data are read. After training, verify all 120 complete histories,
then render the shared-every-epoch pooling/loss plots.
"""
from __future__ import print_function

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT_DEFAULT = "outputs/text/stsb_fixed20_no_earlystop_v1"
OLD_ROOT_NAMES = (
    "stsb_soft_spearman_full_3models_t005_w005",
    "stsb_pure_softsp_cosine_v1",
    "stsb_classic_ltr_pure_cosine_v1",
    "stsb_all_models_ltr_plots_v1",
)
MODELS = ("mean", "flag_zero", "flag")
CLASSIC = ("ranknet", "hinge_fixed", "hinge_label", "lambda_ndcg",
           "listnet_kl", "grouped_listmle")
METHODS = ("mse", "pure_softsp_cosine") + CLASSIC
SEEDS = range(5)
EPOCHS = 20


def source_dir(root, model, method, seed):
    if method == "mse":
        return root / "mse" / model / "mse_scratch" / ("seed_%d" % seed)
    if method == "pure_softsp_cosine":
        return root / "softsp" / model / method / ("seed_%d" % seed)
    return root / "classic" / model / method / ("seed_%d" % seed)


def verify_all(root):
    """Require *every* method / model / seed to have epochs 0..20.

    Epoch 0 is validation-only; the 20 actual training epochs must each contain
    a finite objective, sampled train Spearman and validation Spearman.
    """
    complete = 0
    for model in MODELS:
        for method in METHODS:
            for seed in SEEDS:
                folder = source_dir(root, model, method, seed)
                mp = folder / "metrics.json"
                hp = folder / "history.csv"
                ck = folder / "best_model.pt"
                for p in (mp, hp, ck):
                    if not p.is_file():
                        raise FileNotFoundError(str(p))
                info = json.loads(mp.read_text(encoding="utf-8"))
                if (info.get("model") != model or info.get("seed") != seed
                        or info.get("epochs_ran") != EPOCHS
                        or info.get("max_epochs") != EPOCHS
                        or info.get("min_epochs") != EPOCHS
                        or info.get("patience") != EPOCHS + 1):
                    raise RuntimeError("Incorrect full-20 training settings: %s" % mp)
                if method == "mse":
                    if info.get("variant") != "mse_scratch":
                        raise RuntimeError("Incorrect MSE variant: %s" % mp)
                elif method != "pure_softsp_cosine":
                    if info.get("method") != method:
                        raise RuntimeError("Incorrect LTR method: %s" % mp)
                if info.get("test_evaluated") is not False:
                    raise RuntimeError("Unexpected test access: %s" % mp)
                with hp.open("r", newline="", encoding="utf-8") as fp:
                    history = list(csv.DictReader(fp))
                observed = [int(row["epoch"]) for row in history]
                if observed != list(range(EPOCHS + 1)):
                    raise RuntimeError(
                        "Missing/duplicate epochs for %s: %s" % (hp, observed)
                    )
                loss_key = "train_mse" if method == "mse" else "train_rank_loss"
                val_key = "val_sp" if method == "mse" else "val_spearman"
                for row in history[1:]:
                    for field in (loss_key, "train_sp_sample", val_key):
                        try:
                            value = float(row[field])
                        except (KeyError, ValueError, TypeError):
                            raise RuntimeError("%s missing %s at epoch %s" %
                                               (hp, field, row["epoch"]))
                        if not (-float("inf") < value < float("inf")):
                            raise RuntimeError("%s has nonfinite %s" % (hp, field))
                complete += 1
    if complete != 120:
        raise RuntimeError("Expected 120 results, got %s" % complete)
    print("[FIXED20_VERIFIED] 120/120, each contains 20 train epochs, "
          "5/5 seeds for each model/loss; Test untouched.", flush=True)


def commands(root):
    common = ["--min_epochs", "20", "--patience", "21"]
    roots = {
        "mse": str(root / "mse"),
        "softsp": str(root / "softsp"),
        "classic": str(root / "classic"),
    }
    return [
        ("MSE 15 runs",
         [sys.executable, "-u", "text_repro/run_stsb_soft_spearman_full.py",
          "--variants", "mse_scratch", "--epochs", "20",
          "--output_root", roots["mse"]] + common),
        ("Pure Soft Spearman 15 runs",
         [sys.executable, "-u", "text_repro/run_stsb_pure_softspearman.py",
          "--max_epochs", "20", "--output_root", roots["softsp"],
          "--reference_root", roots["mse"]] + common),
        ("Six classical ranking losses 90 runs",
         [sys.executable, "-u", "text_repro/run_stsb_classic_ltr_suite.py",
          "--max_epochs", "20", "--output_root", roots["classic"],
          "--reference_root", roots["mse"],
          "--softsp_root", roots["softsp"]] + common),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_root", type=Path, default=Path(ROOT_DEFAULT))
    p.add_argument("--dry_run", action="store_true",
                   help="Validate the three training commands without writing/training.")
    p.add_argument("--verify_only", action="store_true",
                   help="Check that all 120 histories have epochs 0..20.")
    p.add_argument("--plot_only", action="store_true",
                   help="Verify and generate plots without training.")
    args = p.parse_args()
    root = args.output_root
    if not root.is_absolute():
        root = REPO / root
    root = root.resolve()
    for forbidden in OLD_ROOT_NAMES:
        if forbidden in root.parts:
            p.error("Never write inside a previous experiment: %s" % root)
    if root == REPO or root == (REPO / "outputs").resolve():
        p.error("Select a dedicated new output directory")
    if sum(bool(x) for x in (args.dry_run, args.verify_only, args.plot_only)) > 1:
        p.error("Choose only one mode")

    if not (args.dry_run or args.plot_only or args.verify_only):
        for stage, command in commands(root):
            print("\n[RUN] %s\n%s" % (stage, " ".join(command)), flush=True)
            subprocess.run(command, cwd=str(REPO), check=True)

    if args.dry_run:
        for stage, cmd in commands(root):
            print("\n[CHECK] %s\n%s" % (stage, " ".join(cmd)), flush=True)
            subprocess.run(cmd + ["--dry_run"], cwd=str(REPO), check=True)
        print("[DRY_RUN_PASS] No training and no outputs written.", flush=True)
        return

    verify_all(root)
    if args.verify_only:
        return

    plot_command = [
        sys.executable, "-u", "text_repro/plot_stsb_fixed20_pooling_comparison.py",
        "--input_root", str(root),
    ]
    print("\n[PLOT] %s" % " ".join(plot_command), flush=True)
    subprocess.run(plot_command, cwd=str(REPO), check=True)
    print("\n[DONE] Full 20-epoch experiment and figures: %s" % root, flush=True)


if __name__ == "__main__":
    main()
