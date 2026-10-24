#!/usr/bin/env python3
"""Export *small* STSB experiment evidence to a reviewable Git tree.

Copies JSON/CSV/TXT metadata and histories from selected result roots; NEVER
copies model checkpoints, large binaries, datasets, or optimizer states.

By default exports:
  * v2 MSE/LTR experiment (20 epochs)
  * joint MSE / hybrid SoftSpearman (5 seeds, 3 architectures)
  * pure SoftSpearman (5 seeds, 3 architectures)

Use --extra-root PATH for the user's earlier "FLaG beats Mean" experiments.
Only after inspecting the generated files, git add/push the export folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

DEFAULT_ROOTS = (
    "outputs/text/stsb_ltr_v2_e20_min12_pat5",
    "outputs/text/stsb_soft_spearman_full_3models_t005_w005",
    "outputs/text/stsb_pure_softsp_cosine_v1",
)
ALLOWED = {"history.csv", "metrics.json", "result.json", "summary.csv",
           "summary_val.csv", "paired_val_differences.csv",
           "pure_summary_val.csv", "pure_paired_vs_mse_val.csv",
           "best_metrics.json", "results.json"}


def sanitize(data):
    """Keep only the evidence needed for train/validation protocol audit."""
    if isinstance(data, dict):
        return {
            key: sanitize(value) for key, value in data.items()
            if not key.lower().startswith("test")
            and not key.lower().startswith("final_test")
        }
    if isinstance(data, list):
        return [sanitize(item) for item in data]
    return data


def hash_text(data: bytes):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("research_audit_exports/stsb_ltr_reconciliation"),
    )
    parser.add_argument(
        "--extra-root", type=Path, action="append", default=[],
        help="Output root of an earlier 'FLaG > Mean' experiment; may repeat.",
    )
    args = parser.parse_args()

    roots = [Path(r) for r in DEFAULT_ROOTS] + args.extra_root
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Export already exists; refuses overwrite: {out}")
    if any(out == x.resolve() or out in x.resolve().parents or x.resolve() in out.parents
           for x in roots):
        parser.error("The export directory must be separate from ALL experiment roots")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment_roots": [], "copied": []}
    n = 0

    for root_idx, root in enumerate(roots):
        name = f"{root_idx:02d}_{root.name}"
        manifest["experiment_roots"].append({
            "label": name,
            "path": str(root),
            "exists": root.is_dir(),
        })
        if not root.is_dir():
            print(f"[MISSING root] {root}")
            continue

        for source in sorted(root.rglob("*")):
            if not source.is_file() or source.name not in ALLOWED:
                continue
            relative = source.relative_to(root)
            target = out / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".json":
                content = json.dumps(
                    sanitize(json.loads(source.read_text(encoding="utf-8"))),
                    indent=2,
                    ensure_ascii=False,
                ) + "\n"
            elif source.suffix == ".csv":
                # Some older history files contain test columns: strip them.
                with source.open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames is None:
                        continue
                    columns = [
                        col for col in reader.fieldnames
                        if col is not None and not col.lower().startswith("test")
                    ]
                    lines = [",".join(columns)]
                    # csv.writer preserves escaped commas/newlines in cells.
                    import io
                    buf = io.StringIO()
                    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
                    writer.writeheader()
                    for row in reader:
                        writer.writerow(row)
                    content = buf.getvalue()
                del lines
            else:
                # Never auto-export arbitrarily large logs.
                if source.stat().st_size > 512_000:
                    print(f"[SKIP large text] {source}")
                    continue
                content = source.read_text(encoding="utf-8")
            target.write_text(content, encoding="utf-8")
            n += 1
            manifest["copied"].append({
                "src": str(source),
                "export": str(target.relative_to(out)),
                "sha256_export": hash_text(content.encode("utf-8")),
                "size_bytes": len(content.encode("utf-8")),
            })

    try:
        manifest["git_commit_on_server"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        manifest["git_commit_on_server"] = None
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] exported {n} metadata/history files into {out}")
    print("[NOTE] No *.pt checkpoints or datasets copied. Test metrics sanitized.")
    print(f"[NEXT] git add -f {args.out}")
    print("[NEXT] git commit -m 'Export STSB train/val evidence for protocol audit'")
    print("[NEXT] git push origin FLaG-STFT-mechanism")


if __name__ == "__main__":
    main()
