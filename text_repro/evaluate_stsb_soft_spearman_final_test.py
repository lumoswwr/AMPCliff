#!/usr/bin/env python3
"""One protocol-frozen final STSB test readout, without training.

Loads ONLY previously selected best_model.pt checkpoints from the completed
5-seed x 3-architecture x 2-objective validation-only study. Does not modify
source results, reselect checkpoints, compute gradients or tune hyperparameters.
Writes a separate final evaluation record and paired-difference reports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_author_text_protocol import PairCollator, PairDataset, seed_everything
from run_stsb_ltr_v2 import evaluate, make_model

MODELS = ("mean", "flag_zero", "flag")
VARIANTS = ("mse_scratch", "mse_softsp_w005")
SEEDS = tuple(range(5))
PROTOCOL = "softsp_3models_v1_val_only"
FIXED_CONFIG = {
    "max_epochs": 20,
    "min_epochs": 12,
    "patience": 5,
    "rank_temperature": 0.05,
    "encoder_lr": 1e-5,
    "new_lr": 1e-3,
}
HEADER = [
    "model", "variant", "seed", "best_epoch",
    "source_val_spearman", "recomputed_val_spearman",
    "test_spearman", "test_pearson", "test_native_mse",
    "checkpoint_sha256", "checkpoint_path",
]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def check_inputs(root: Path):
    inputs = []
    for model in MODELS:
        for variant in VARIANTS:
            for seed in SEEDS:
                directory = root / model / variant / f"seed_{seed}"
                config_path = directory / "metrics.json"
                ckpt_path = directory / "best_model.pt"
                history_path = directory / "history.csv"
                if not (config_path.is_file() and ckpt_path.is_file()
                        and history_path.is_file()):
                    raise FileNotFoundError(
                        f"Missing metrics, checkpoint or history for {model}/{variant}/seed_{seed}: {directory}"
                    )
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                required = {
                    "protocol": PROTOCOL,
                    "model": model, "variant": variant, "seed": seed,
                    "rank_weight": (0.0 if variant == "mse_scratch" else 0.05),
                    **FIXED_CONFIG,
                }
                for key, expected in required.items():
                    if cfg.get(key) != expected:
                        raise RuntimeError(
                            f"Checkpoint protocol mismatch in {config_path}: "
                            f"{key}={cfg.get(key)!r}, expected={expected!r}"
                        )
                if cfg.get("test_evaluated") is not False:
                    raise RuntimeError(
                        f"Source metrics unexpectedly indicate test evaluation: {config_path}"
                    )
                inputs.append((model, variant, seed, ckpt_path, cfg))
    if len(inputs) != 30:
        raise RuntimeError(f"Expected 30 runs, got {len(inputs)}")
    return inputs


def write_csv(path: Path, records, columns=None):
    if not records:
        raise ValueError("Cannot write empty result")
    fieldnames = columns or list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def mean_sd(xs):
    return statistics.mean(xs), statistics.stdev(xs)


def summarize(records, out):
    by_key = {
        (r["model"], r["variant"], r["seed"]): r
        for r in records
    }
    rows, paired = [], []
    for model in MODELS:
        for variant in VARIANTS:
            selected = [
                by_key[(model, variant, seed)] for seed in SEEDS
            ]
            metrics = {}
            for key in ("test_spearman", "test_pearson", "test_native_mse"):
                vals = [float(x[key]) for x in selected]
                avg, sd = mean_sd(vals)
                metrics[key + "_mean"] = avg
                metrics[key + "_sample_sd"] = sd
            rows.append({"model": model, "variant": variant, **metrics})

        for name, baseline in (
            ("softsp_minus_own_mse", [
                (model, "mse_softsp_w005", seed, model, "mse_scratch", seed)
                for seed in SEEDS
            ]),
            ("softsp_minus_mean_mse", [
                (model, "mse_softsp_w005", seed, "mean", "mse_scratch", seed)
                for seed in SEEDS
            ]),
        ):
            for metric in ("test_spearman", "test_pearson", "test_native_mse"):
                diffs = [
                    float(by_key[(m, v, s)][metric]) -
                    float(by_key[(bm, bv, bs)][metric])
                    for m, v, s, bm, bv, bs in baseline
                ]
                avg, sd = mean_sd(diffs)
                paired.append({
                    "model": model, "comparison": name, "metric": metric,
                    "mean_delta": avg, "sample_sd_delta": sd,
                    "positive_seed_count": sum(x > 0 for x in diffs),
                    "negative_seed_count": sum(x < 0 for x in diffs),
                    "per_seed_deltas": json.dumps(diffs),
                })
    write_csv(out / "summary_test.csv", rows)
    write_csv(out / "paired_test_differences.csv", paired)

    print("\n[FINAL_TEST_SUMMARY] Frozen checkpoints, no post-test tuning.", flush=True)
    for row in rows:
        print(f"{row['model']:10} {row['variant']:18} "
              f"test_sp={row['test_spearman_mean']:.6f} ± {row['test_spearman_sample_sd']:.6f} "
              f"test_pe={row['test_pearson_mean']:.6f} "
              f"test_native_mse={row['test_native_mse_mean']:.6f}", flush=True)
    print("\n[PAIRED_TEST_SPEARMAN_DELTAS]", flush=True)
    for row in paired:
        if row["metric"] != "test_spearman":
            continue
        print(f"{row['model']:10} {row['comparison']:24} "
              f"mean_delta={row['mean_delta']:+.6f} "
              f"sd={row['sample_sd_delta']:.6f} "
              f"positive_seeds={row['positive_seed_count']}/5 "
              f"per_seed={row['per_seed_deltas']}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint_root", type=Path,
        default=Path("outputs/text/stsb_soft_spearman_full_3models_t005_w005")
    )
    p.add_argument(
        "--output_root", type=Path,
        default=Path("outputs/text/stsb_soft_spearman_final_test_frozen_v1")
    )
    p.add_argument("--model_path", default="/home/data/home/wwr_lumos/models/roberta-base")
    p.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if args.batch_size < 1:
        p.error("--batch_size must be >=1")
    source = args.checkpoint_root.resolve()
    output = args.output_root.resolve()
    if output == source or source in output.parents or output in source.parents:
        p.error("Output must be separate from source checkpoint tree")
    inputs = check_inputs(args.checkpoint_root)
    print("[preflight] 30 checkpoint+config+history bundles verified.", flush=True)
    print(f"[preflight] source={args.checkpoint_root}", flush=True)
    print(f"[preflight] final_outputs={args.output_root}", flush=True)
    if args.dry_run:
        print("[dry_run] Does not read test split or create outputs.", flush=True)
        return

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing final-test directory: {args.output_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_from_disk(args.stsb_data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    collator = PairCollator(tokenizer, max_length=128, label_dtype=torch.float32)
    val_dl = DataLoader(PairDataset(raw["validation"], "stsb"),
                        batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_dl = DataLoader(PairDataset(raw["test"], "stsb"),
                         batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    results = []
    for n, (model_name, variant, seed, checkpoint, cfg) in enumerate(inputs, start=1):
        seed_everything(seed)
        args.model = model_name
        model = make_model(model_name, args, device)
        model.load_state_dict(read_checkpoint(checkpoint), strict=True)
        val_metrics, _ = evaluate(model, val_dl, device)
        expected_val = float(cfg["best_val_spearman"])
        if not math.isfinite(val_metrics["spearman"]) or abs(
            val_metrics["spearman"] - expected_val
        ) > 1e-4:
            raise RuntimeError(
                f"Checkpoint validation mismatch: {checkpoint}, "
                f"saved={expected_val}, reproduced={val_metrics['spearman']}"
            )
        test_metrics, _ = evaluate(model, test_dl, device)
        results.append({
            "model": model_name, "variant": variant, "seed": seed,
            "best_epoch": int(cfg["best_epoch"]),
            "source_val_spearman": expected_val,
            "recomputed_val_spearman": val_metrics["spearman"],
            "test_spearman": test_metrics["spearman"],
            "test_pearson": test_metrics["pearson"],
            "test_native_mse": test_metrics["native_mse"],
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_path": str(checkpoint),
        })
        print(f"[{n:02}/30] {model_name}/{variant}/s{seed} "
              f"val_sp={val_metrics['spearman']:.6f} "
              f"test_sp={test_metrics['spearman']:.6f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(args.output_root / "per_seed_test.csv", results, HEADER)
    summarize(results, args.output_root)
    manifest = {
        "protocol": "STSB validation-selected fixed Soft Spearman 3x2x5 final test",
        "checkpoint_root": str(args.checkpoint_root),
        "num_conditions": 30,
        "checkpoint_selection": "validation Spearman only, fixed before test",
        "test_usage": "One final evaluation of all 30 predeclared conditions",
        "no_training": True,
        "model_path": args.model_path,
        "dataset_path": args.stsb_data_path,
        "batch_size": args.batch_size,
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "variants": list(VARIANTS),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
