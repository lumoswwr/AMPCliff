#!/usr/bin/env python3
"""
Compare Sprint DC-only layer curves before and after backbone fine-tuning.

Conditions:
  1) pretrained/frozen RoBERTa-base
  2) Sprint Mean-pooling task-finetuned RoBERTa checkpoints

For every hidden representation H0..H12:
  masked mean (DC/global mean) -> pair cosine -> Average Precision.

The final layer of a Mean-pooling fine-tuned checkpoint should reproduce that
checkpoint's cosine AP, because the cosine-logit head has positive scale and
therefore preserves ranking.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from transformers import AutoModel, AutoTokenizer

from probe_dc_layer_curve import (
    collect_layer_cosines,
    load_stsb_backbone_checkpoint,
    make_loader,
    seed_everything,
)


def infer_seed(path: Path, fallback: int) -> int:
    m = re.search(r"seed[_-](\d+)", str(path))
    return int(m.group(1)) if m else fallback


@torch.no_grad()
def probe_model(
    model,
    loader,
    device: torch.device,
    *,
    condition: str,
    checkpoint: str,
    seed,
) -> pd.DataFrame:
    model.to(device)
    model.eval()

    layer_cosines, labels = collect_layer_cosines(
        model,
        loader,
        device,
    )
    labels_np = np.asarray(labels, dtype=np.int64)

    rows = []
    for layer, cosine in enumerate(layer_cosines):
        ap = float(
            average_precision_score(
                labels_np,
                np.asarray(cosine, dtype=np.float64),
            )
        )
        rows.append(
            {
                "condition": condition,
                "checkpoint": checkpoint,
                "seed": seed,
                "layer": layer,
                "layer_name": (
                    "Embedding"
                    if layer == 0
                    else f"Layer {layer}"
                ),
                "average_precision": ap,
            }
        )

    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    out = (
        raw.groupby(
            ["condition", "layer", "layer_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            ap_mean=("average_precision", "mean"),
            ap_std=("average_precision", "std"),
            n=("checkpoint", "nunique"),
        )
        .sort_values(["condition", "layer"])
        .reset_index(drop=True)
    )
    out["ap_std"] = out["ap_std"].fillna(0.0)
    return out


def make_delta(summary: pd.DataFrame) -> pd.DataFrame:
    frozen = (
        summary[
            summary["condition"] == "pretrained_frozen"
        ][["layer", "layer_name", "ap_mean"]]
        .rename(columns={"ap_mean": "frozen_ap"})
    )

    ft = (
        summary[
            summary["condition"] == "mean_finetuned"
        ][["layer", "layer_name", "ap_mean", "ap_std"]]
        .rename(
            columns={
                "ap_mean": "finetuned_ap",
                "ap_std": "finetuned_ap_std",
            }
        )
    )

    out = frozen.merge(
        ft,
        on=["layer", "layer_name"],
        how="outer",
    )
    out["finetuned_minus_frozen_ap"] = (
        out["finetuned_ap"] - out["frozen_ap"]
    )
    return out.sort_values("layer").reset_index(drop=True)


def plot(summary: pd.DataFrame, delta: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))

    ax = axes[0]
    for condition, label in [
        ("pretrained_frozen", "Pretrained/frozen RoBERTa"),
        ("mean_finetuned", "Sprint Mean-trained RoBERTa"),
    ]:
        sub = (
            summary[summary["condition"] == condition]
            .sort_values("layer")
        )
        x = sub["layer"].to_numpy()
        y = sub["ap_mean"].to_numpy()
        std = sub["ap_std"].to_numpy()

        ax.plot(x, y, marker="o", label=label)
        if np.any(std > 0):
            ax.fill_between(
                x,
                y - std,
                y + std,
                alpha=0.15,
            )

    max_layer = int(summary["layer"].max())
    ticks = np.arange(max_layer + 1)
    labels = ["Emb"] + [str(i) for i in range(1, max_layer + 1)]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("RoBERTa representation")
    ax.set_ylabel("Average Precision")
    ax.set_title("Sprint: DC-readable matching signal")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax.plot(
        delta["layer"],
        delta["finetuned_minus_frozen_ap"],
        marker="o",
    )
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("RoBERTa representation")
    ax.set_ylabel("Delta AP")
    ax.set_title("Effect of Sprint backbone fine-tuning on DC readability")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        out_dir / "sprint_dc_finetune_control.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "sprint_dc_finetune_control.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="Sprint Mean + finetuned-backbone best_model.pt checkpoints.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path(
            "/home/data/home/wwr_lumos/models/roberta-base"
        ),
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path(
            "/home/data/home/wwr_lumos/AMPCliff/"
            "data/text/sprintduplicatequestions_adaptation"
        ),
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "/home/data/home/wwr_lumos/AMPCliff/"
            "outputs/text/sprint_dc_finetune_control"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    for ckpt in args.checkpoints:
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        local_files_only=True,
    )

    loader, n_pairs = make_loader(
        args.data_path,
        args.split,
        target_key="label",
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    print("device:", device)
    print("split:", args.split)
    print("pairs:", n_pairs)

    frozen_model = AutoModel.from_pretrained(
        str(args.model_path),
        local_files_only=True,
    )
    frozen_df = probe_model(
        frozen_model,
        loader,
        device,
        condition="pretrained_frozen",
        checkpoint="pretrained_roberta_base",
        seed=np.nan,
    )
    del frozen_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    frames: List[pd.DataFrame] = [frozen_df]
    metas: List[Dict[str, object]] = []

    for idx, ckpt in enumerate(args.checkpoints):
        model = AutoModel.from_pretrained(
            str(args.model_path),
            local_files_only=True,
        )
        meta = load_stsb_backbone_checkpoint(model, ckpt)
        seed = infer_seed(ckpt, idx)

        frame = probe_model(
            model,
            loader,
            device,
            condition="mean_finetuned",
            checkpoint=str(ckpt),
            seed=seed,
        )
        frames.append(frame)
        metas.append(
            {
                "checkpoint": str(ckpt),
                "seed": seed,
                **meta,
            }
        )

        final_ap = float(
            frame.sort_values("layer")
            .iloc[-1]["average_precision"]
        )
        print(
            f"[Sprint FT seed={seed}] final-layer "
            f"DC/Mean AP={final_ap:.6f}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw = pd.concat(frames, ignore_index=True)
    summary = summarize(raw)
    delta = make_delta(summary)

    raw.to_csv(
        args.output_dir / "sprint_dc_finetune_raw.csv",
        index=False,
    )
    summary.to_csv(
        args.output_dir / "sprint_dc_finetune_summary.csv",
        index=False,
    )
    delta.to_csv(
        args.output_dir / "sprint_dc_finetune_delta.csv",
        index=False,
    )

    plot(summary, delta, args.output_dir)

    metadata = {
        "probe": "Sprint frozen vs Mean-finetuned backbone DC layer curve",
        "split": args.split,
        "pairs": n_pairs,
        "dc_definition": (
            "masked mean over all valid tokens including special tokens"
        ),
        "checkpoints": metas,
        "important": (
            "The fine-tuned condition is a mechanism control, not the "
            "published Sprint frozen-backbone protocol."
        ),
    }
    with open(
        args.output_dir / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nFinal-layer summary:")
    last_layer = int(summary["layer"].max())
    print(
        summary[summary["layer"] == last_layer]
        .to_string(index=False)
    )
    print("\nSaved to:", args.output_dir)


if __name__ == "__main__":
    main()
