#!/usr/bin/env python3
"""Final TEST inference from 120 EXISTING STSB validation-selected checkpoints.

Models: Mean / FLaG-Zero / FLaG
Methods: MSE, pure Soft Spearman, and 6 corrected classical ranking objectives
Seeds: 0...4 = 120 previously trained checkpoints.

This script NEVER trains, changes or reselects checkpoints. It reproduces
Validation Spearman first (tolerance 1e-4), then tests the selected checkpoint
on the pre-existing official STS-B test split. No changes to source runs.
Separate output tree with resumable per-checkpoint results and paired reports.

NOTE: STS-B test has previously been seen during other research experiments,
so this is a frozen-protocol test comparison, not a pristine blind holdout.
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

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_author_text_protocol import PairDataset, PairCollator, seed_everything
from run_stsb_ltr_v2 import evaluate, make_model
from stsb_classic_ltr_losses import METHODS as CLASSIC_METHODS

MODELS = ("mean", "flag_zero", "flag")
METHODS = ("mse", "pure_softsp_cosine") + tuple(CLASSIC_METHODS)
SEEDS = tuple(range(5))
EXPECTED = {
    "mse": "softsp_3models_v1_val_only",
    "pure_softsp_cosine": "stsb_pure_softspearman_cosine_v1_val_only",
    **{name: "stsb_classic_ltr_fixed_temperature_v1_val_only"
       for name in CLASSIC_METHODS},
}


def run_directory(args, model, method, seed):
    if method == "mse":
        root, subdir = args.mse_root, "mse_scratch"
    elif method == "pure_softsp_cosine":
        root, subdir = args.softsp_root, "pure_softsp_cosine"
    else:
        root, subdir = args.classic_root, method
    return root / model / subdir / f"seed_{seed}"


def build_inputs(args):
    records = []
    for model in MODELS:
        for method in METHODS:
            for seed in SEEDS:
                folder = run_directory(args, model, method, seed)
                met_file = folder / "metrics.json"
                history_file = folder / "history.csv"
                checkpoint = folder / "best_model.pt"
                for file in (met_file, history_file, checkpoint):
                    if not file.is_file():
                        raise FileNotFoundError(
                            f"Missing source result for {model}/{method}/s{seed}: {file}"
                        )
                info = json.loads(met_file.read_text(encoding="utf-8"))
                for name, expect in (
                    ("model", model),
                    ("seed", seed),
                    ("protocol", EXPECTED[method]),
                ):
                    if info.get(name) != expect:
                        raise RuntimeError(
                            f"Invalid checkpoint metadata {met_file}: "
                            f"{name}={info.get(name)!r} expected={expect!r}"
                        )
                if method == "mse" and info.get("variant") != "mse_scratch":
                    raise RuntimeError(f"Incorrect MSE reference: {met_file}")
                if method in CLASSIC_METHODS and info.get("method") != method:
                    raise RuntimeError(f"Incorrect LTR method: {met_file}")
                if method == "pure_softsp_cosine" and info.get("loss") != "pure_soft_midrank_spearman_on_cosine":
                    raise RuntimeError(f"Incorrect pure SoftSP reference: {met_file}")
                if info.get("test_evaluated") is not False:
                    raise RuntimeError(f"Source must be validation-only: {met_file}")
                target = float(info["best_val_spearman"])
                if not math.isfinite(target):
                    raise RuntimeError(f"Nonfinite saved validation Spearman: {met_file}")
                records.append({
                    "model": model, "method": method, "seed": seed,
                    "checkpoint": checkpoint, "source_metrics": met_file,
                    "saved_val_sp": target, "best_epoch": int(info["best_epoch"]),
                })
    assert len(records) == 120
    return records


def load_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_file(out, item):
    return out / "per_checkpoint" / item["model"] / item["method"] / f"seed_{item['seed']}.json"


def validate_existing_result(item, result):
    expected = {
        "model": item["model"], "method": item["method"],
        "seed": item["seed"], "best_epoch": item["best_epoch"],
        "source_checkpoint": str(item["checkpoint"]),
        "checkpoint_size": item["checkpoint"].stat().st_size,
        "checkpoint_mtime_ns": item["checkpoint"].stat().st_mtime_ns,
        "source_val_sp": item["saved_val_sp"],
    }
    for key, val in expected.items():
        if result.get(key) != val:
            raise RuntimeError(
                f"Saved TEST record no longer matches source: "
                f"{item['model']}/{item['method']}/seed_{item['seed']} "
                f"{key}: existing={result.get(key)!r}, expected={val!r}"
            )
    if not all(math.isfinite(float(result[k])) for k in (
        "reproduced_val_sp", "test_sp", "test_pe"
    )):
        raise RuntimeError(f"Nonfinite saved TEST record: {result}")
    if abs(float(result["reproduced_val_sp"]) - item["saved_val_sp"]) > 1e-4:
        raise RuntimeError("Saved validation reproduction no longer agrees")
    return result


def summary(inputs, out):
    results = []
    for item in inputs:
        path = result_file(out, item)
        if not path.is_file():
            print(f"[partial] missing {path}; complete summary deferred", flush=True)
            return
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_existing_result(item, record)
        results.append(record)

    by_key = {(r["model"], r["method"], r["seed"]): r for r in results}
    groups, pairs = [], []
    for method in METHODS:
        for model in MODELS:
            subset = [by_key[(model, method, seed)] for seed in SEEDS]
            vals = [float(row["test_sp"]) for row in subset]
            groups.append({
                "method": method, "model": model, "n_seeds": len(vals),
                "mean_test_sp": statistics.mean(vals),
                "sd_test_sp": statistics.stdev(vals),
                "mean_test_pe": statistics.mean(float(row["test_pe"]) for row in subset),
                "mean_val_sp": statistics.mean(float(row["source_val_sp"]) for row in subset),
                "per_seed_test_sp": json.dumps(vals),
            })
        for model in MODELS:
            for label, baseline in (
                ("vs_own_mse", (model, "mse")),
                ("vs_mean_same_loss", ("mean", method)),
                ("vs_mean_mse", ("mean", "mse")),
            ):
                deltas = [
                    float(by_key[(model, method, seed)]["test_sp"]) -
                    float(by_key[(baseline[0], baseline[1], seed)]["test_sp"])
                    for seed in SEEDS
                ]
                pairs.append({
                    "method": method, "model": model, "comparison": label,
                    "mean_delta": statistics.mean(deltas),
                    "sd_delta": statistics.stdev(deltas),
                    "positive_seeds": sum(x > 0 for x in deltas),
                    "per_seed_deltas": json.dumps(deltas),
                })

    save_csv(out / "per_seed_test.csv", results)
    save_csv(out / "summary_test.csv", groups)
    save_csv(out / "paired_test.csv", pairs)

    print("\n[CLASSIC_LTR_FINAL_TEST] 120/120 frozen checkpoints evaluated; no retraining", flush=True)
    print("method                   Mean TEST SP          FLaG-Zero TEST SP      FLaG TEST SP", flush=True)
    for method in METHODS:
        entries = [next(r for r in groups if r["method"] == method and r["model"] == model) for model in MODELS]
        vals = "  ".join(f"{x['mean_test_sp']:.6f} ±{x['sd_test_sp']:.6f}" for x in entries)
        print(f"{method:23} {vals}", flush=True)

    print("\n[CLASSIC_LTR_TEST_ARCHITECTURE] Each loss: FLaG minus Mean on matching seed", flush=True)
    for method in METHODS:
        for model in ("flag_zero", "flag"):
            r = next(r for r in pairs if r["method"] == method and r["model"] == model
                     and r["comparison"] == "vs_mean_same_loss")
            print(f"{method:23} {model:10} delta={r['mean_delta']:+.6f} "
                  f"sd={r['sd_delta']:.6f} improved={r['positive_seeds']}/5", flush=True)

    print("\n[CLASSIC_LTR_TEST_VS_MSE] Paired, same model and seed", flush=True)
    for method in METHODS:
        if method == "mse":
            continue
        for model in MODELS:
            r = next(r for r in pairs if r["method"] == method and r["model"] == model
                     and r["comparison"] == "vs_own_mse")
            print(f"{method:23} {model:10} delta={r['mean_delta']:+.6f} "
                  f"sd={r['sd_delta']:.6f} improved={r['positive_seeds']}/5", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mse_root", type=Path,
                   default=Path("outputs/text/stsb_soft_spearman_full_3models_t005_w005"))
    p.add_argument("--softsp_root", type=Path,
                   default=Path("outputs/text/stsb_pure_softsp_cosine_v1"))
    p.add_argument("--classic_root", type=Path,
                   default=Path("outputs/text/stsb_classic_ltr_pure_cosine_v1"))
    p.add_argument("--output_root", type=Path,
                   default=Path("outputs/text/stsb_classic_ltr_final_test_120_frozen_v1"))
    p.add_argument("--model_path",
                   default="/home/data/home/wwr_lumos/models/roberta-base")
    p.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--summary_only", action="store_true")
    args = p.parse_args()

    if args.batch_size <= 0:
        p.error("--batch_size must be positive")
    out = args.output_root.resolve()
    for source in (args.mse_root, args.softsp_root, args.classic_root):
        root = source.resolve()
        if out == root or out in root.parents or root in out.parents:
            p.error("The final Test output root must be completely separate from training roots")
    inputs = build_inputs(args)
    print(f"[preflight] verified {len(inputs)} checkpoint+history+metrics bundles", flush=True)
    print("[preflight] Training and best checkpoint selections already completed.", flush=True)

    if args.dry_run:
        print("[dry_run] no test accessed; no files written", flush=True)
        return
    if args.summary_only:
        summary(inputs, args.output_root)
        return

    if args.output_root.exists() and any(args.output_root.iterdir()):
        # Resume is permitted ONLY for an existing run with same manifest.
        mf = args.output_root / "manifest.json"
        if not mf.is_file():
            raise FileExistsError(
                f"Nonempty Test directory without matching manifest: {args.output_root}"
            )
        saved = json.loads(mf.read_text(encoding="utf-8"))
        if saved.get("protocol") != "frozen_stsb_120_test_v1" or (
            saved.get("batch_size") != args.batch_size
        ) or saved.get("source_roots") != [
            str(args.mse_root), str(args.softsp_root), str(args.classic_root)
        ]:
            raise RuntimeError("Refusing to reuse mismatched final-test output root")
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "manifest.json").write_text(
            json.dumps({
                "protocol": "frozen_stsb_120_test_v1",
                "number_of_checkpoints": 120,
                "selection": "validation Spearman; preselected during training",
                "test_use": "only final evaluation; never for selection or tuning",
                "note": "STSB test previously consulted in historical experiments, not a pristine blind holdout.",
                "batch_size": args.batch_size,
                "source_roots": [
                    str(args.mse_root), str(args.softsp_root), str(args.classic_root)
                ],
                "model_path": args.model_path,
                "stsb_data_path": args.stsb_data_path,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    seed_everything(0)
    data = load_from_disk(args.stsb_data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    collate = PairCollator(tokenizer, 128, torch.float32)
    val_loader = DataLoader(PairDataset(data["validation"], "stsb"),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate)
    test_loader = DataLoader(PairDataset(data["test"], "stsb"),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, item in enumerate(inputs, start=1):
        dest = result_file(args.output_root, item)
        if dest.is_file():
            record = json.loads(dest.read_text(encoding="utf-8"))
            validate_existing_result(item, record)
            print(f"[skip {i:03d}/120] {item['model']}/{item['method']}/s{item['seed']}", flush=True)
            continue
        if dest.exists():
            raise RuntimeError(f"Unexpected existing result file: {dest}")

        seed_everything(item["seed"])
        model = make_model(item["model"], args, device)
        model.load_state_dict(load_state(item["checkpoint"]), strict=True)
        vm, _ = evaluate(model, val_loader, device)
        if not math.isfinite(vm["spearman"]) or abs(
            vm["spearman"] - item["saved_val_sp"]
        ) > 1e-4:
            raise RuntimeError(
                f"VALIDATION MISMATCH before test: {item['model']}/{item['method']}/s{item['seed']} "
                f"saved={item['saved_val_sp']:.8f}, reproduced={vm['spearman']:.8f}; "
                "check checkpoint/model/dataset/eval batch before proceeding"
            )
        tm, _ = evaluate(model, test_loader, device)
        if not math.isfinite(tm["spearman"]):
            raise RuntimeError(f"Nonfinite Test Spearman: {item}")
        stat = item["checkpoint"].stat()
        record = {
            "model": item["model"], "method": item["method"],
            "seed": item["seed"], "best_epoch": item["best_epoch"],
            "source_val_sp": item["saved_val_sp"],
            "reproduced_val_sp": vm["spearman"],
            "test_sp": tm["spearman"],
            "test_pe": tm["pearson"],
            "checkpoint_size": stat.st_size,
            "checkpoint_mtime_ns": stat.st_mtime_ns,
            "source_checkpoint": str(item["checkpoint"]),
        }
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"[{i:03d}/120] {item['model']}/{item['method']}/s{item['seed']} "
            f"VAL={vm['spearman']:.6f} TEST={tm['spearman']:.6f}",
            flush=True,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary(inputs, args.output_root)


if __name__ == "__main__":
    main()
