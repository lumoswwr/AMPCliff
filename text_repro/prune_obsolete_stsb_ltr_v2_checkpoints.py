#!/usr/bin/env python3
"""Safely prune OLD STSB LTR-v2 non-MSE checkpoints after audit export.

Default: READ-ONLY dry-run. With --apply, DELETE only:
outputs/text/stsb_ltr_v2_e20_min12_pat5/{mean,flag_zero,flag}/
{ranknet,ranknet_scaled,pairwise_hinge,listnet,listmle,lambdarank}/
seed_{0..4}/best_model.pt

Never deletes any directory, code, MSE weights, current corrected LTR runs,
Pure Soft Spearman, Hybrid runs, frozen-encoder mechanistic runs, or logs.

Safety: require completed old metrics/history AND a matching train/validation
metrics.json snapshot previously exported into GitHub's
research_audit_exports/stsb_ltr_reconciliation/00_stsb_ltr_v2_e20_min12_pat5.
No broad globs, recursive deletion or accidental symlink traversal.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MODELS = ("mean", "flag_zero", "flag")
OLD_LOSSES = (
    "ranknet", "ranknet_scaled", "pairwise_hinge",
    "listnet", "listmle", "lambdarank",
)
SEEDS = tuple(range(5))
OLD_ROOT = Path("outputs/text/stsb_ltr_v2_e20_min12_pat5")
AUDIT_ROOT = Path(
    "research_audit_exports/stsb_ltr_reconciliation/"
    "00_stsb_ltr_v2_e20_min12_pat5"
)


def gib(number):
    return number / (1024 ** 3)


def path_is_inside(path: Path, root: Path) -> bool:
    """Compatible with Python 3.7/3.8 (Path.is_relative_to needs Python 3.9)."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def same_record(original: dict, exported: dict, model: str, loss: str, seed: int):
    for doc in (original, exported):
        if doc.get("model") != model or doc.get("loss") != loss or doc.get("seed") != seed:
            return False
        if doc.get("protocol_version") != "stsb_ltr_v2_equal_budget":
            return False
    for key in ("best_val_spearman", "best_epoch", "max_epochs", "min_epochs", "patience"):
        if original.get(key) != exported.get(key):
            return False
    return True


def scan(repo_root: Path):
    existing, skipped = [], []
    old_root = repo_root / OLD_ROOT
    audit_root = repo_root / AUDIT_ROOT
    for model in MODELS:
        for loss in OLD_LOSSES:
            for seed in SEEDS:
                relative = Path(model) / loss / f"seed_{seed}"
                old_dir = old_root / relative
                chk = old_dir / "best_model.pt"
                if not chk.exists() and not chk.is_symlink():
                    continue
                if chk.is_symlink() or not chk.is_file() or old_dir.is_symlink():
                    skipped.append((chk, "symlink or nonregular checkpoint/directory"))
                    continue
                try:
                    if not path_is_inside(chk.resolve(), old_root.resolve()):
                        skipped.append((chk, "checkpoint escapes the approved old root"))
                        continue
                except OSError:
                    skipped.append((chk, "cannot resolve safely"))
                    continue

                original_metrics = old_dir / "metrics.json"
                original_history = old_dir / "history.csv"
                exported_metrics = audit_root / relative / "metrics.json"
                exported_history = audit_root / relative / "history.csv"
                if not all(p.is_file() for p in (
                    original_metrics, original_history, exported_metrics, exported_history
                )):
                    skipped.append((chk, "missing original or backed-up metrics/history"))
                    continue
                if any(p.is_symlink() for p in (
                    original_metrics, original_history, exported_metrics, exported_history
                )):
                    skipped.append((chk, "record symlink not permitted"))
                    continue
                try:
                    original = json.loads(original_metrics.read_text(encoding="utf-8"))
                    exported = json.loads(exported_metrics.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as err:
                    skipped.append((chk, f"cannot parse metrics: {err}"))
                    continue
                if not same_record(original, exported, model, loss, seed):
                    skipped.append((chk, "backup record mismatches original training result"))
                    continue
                existing.append((chk, chk.stat().st_size))
    return existing, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually unlink eligible old non-MSE best_model.pt files."
    )
    parser.add_argument(
        "--repo_root", type=Path,
        default=Path(__file__).resolve().parents[1],
        help="AMPCliff checkout; defaults to this script's repository."
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    if not (repo / "text_repro" / "run_stsb_ltr_v2.py").is_file():
        parser.error("Not an AMPCliff repository: expected text_repro/run_stsb_ltr_v2.py")

    eligible, skipped = scan(repo)
    total = sum(size for _, size in eligible)
    print("[scope] LEGACY stsb_ltr_v2 3 models x 6 old LTR losses x 5 seeds")
    print("[protected] ALL MSE checkpoints, CURRENT 90-run suite, pure SoftSP, "
          "Hybrid, frozen-encoder and older mechanism experiments.")
    print("[protected] metrics.json, history.csv, all code, all logs.")
    print(f"[eligible] {len(eligible)}/90 old LTR-v2 checkpoints; "
          f"recoverable={gib(total):.2f} GiB")
    for path, size in eligible:
        print(f"  {'DELETE' if args.apply else 'WOULD DELETE'} "
              f"{path.relative_to(repo) if path_is_inside(path, repo) else path} "
              f"({gib(size):.3f} GiB)")
    for path, why in skipped:
        print(f"  SKIP {path}: {why}")

    if not args.apply:
        print("\n[DRY RUN] No files modified. Rerun with --apply to remove ONLY eligible files.")
        return
    if not eligible:
        print("\n[NOOP] Nothing matched the strict safety requirements.")
        return

    removed = 0
    removed_bytes = 0
    for path, expected_size in eligible:
        # Recheck immediately before unlink in case a concurrent process
        # replaced any file between scan and delete.
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_size:
            print(f"[changed; skip] {path}")
            continue
        path.unlink()
        removed += 1
        removed_bytes += expected_size
    print(f"\n[DONE] Removed {removed} historical non-MSE checkpoints; "
          f"reclaimed about {gib(removed_bytes):.2f} GiB.")
    print("[DONE] All original metrics/history and audit-export evidence preserved.")


if __name__ == "__main__":
    main()
