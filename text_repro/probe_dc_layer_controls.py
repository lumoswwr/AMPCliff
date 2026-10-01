#!/usr/bin/env python3
"""
Controls for the layer-wise DC-only probe.

This extends probe_dc_layer_curve.py with three STS-B backbone conditions:

1. pretrained/frozen RoBERTa-base
2. STS-B Mean-pooling fine-tuned RoBERTa
3. STS-B FLaG fine-tuned RoBERTa

Sprint remains the protocol-specified frozen RoBERTa control.

The goal is to separate:
- task/data effects,
- backbone fine-tuning effects,
- pooling-dependent representation effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from transformers import AutoModel, AutoTokenizer

from probe_dc_layer_curve import (
    collect_layer_cosines,
    correlation_metrics,
    make_loader,
    probe_sprint,
    probe_stsb_checkpoint,
    seed_everything,
)


@torch.no_grad()
def probe_stsb_frozen(
    model_path: Path,
    loader,
    device: torch.device,
) -> pd.DataFrame:
    model = AutoModel.from_pretrained(
        str(model_path),
        local_files_only=True,
    ).to(device)

    layer_cosines, gold = collect_layer_cosines(
        model,
        loader,
        device,
    )

    rows = []
    for layer, pred in enumerate(layer_cosines):
        metrics = correlation_metrics(pred, gold)
        rows.append(
            {
                "dataset": "stsb",
                "condition": "pretrained_frozen",
                "checkpoint": "pretrained_roberta_base",
                "seed": np.nan,
                "layer": layer,
                "layer_name": (
                    "Embedding"
                    if layer == 0
                    else f"Layer {layer}"
                ),
                "spearman": metrics["spearman"],
                "pearson": metrics["pearson"],
            }
        )

    print(
        "[STSB frozen] final-layer DC/Mean "
        f"Spearman={rows[-1]['spearman']:.6f} "
        f"Pearson={rows[-1]['pearson']:.6f}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def probe_checkpoint_group(
    *,
    condition: str,
    model_path: Path,
    checkpoints: List[Path],
    loader,
    device: torch.device,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    frames = []
    metadata = []

    for checkpoint_index, checkpoint_path in enumerate(checkpoints):
        frame, meta = probe_stsb_checkpoint(
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            loader=loader,
            device=device,
            checkpoint_index=checkpoint_index,
        )
        frame["condition"] = condition
        frames.append(frame)
        metadata.append(meta)

    return pd.concat(frames, ignore_index=True), metadata


def summarize_stsb_controls(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(
            ["condition", "layer", "layer_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            n_checkpoints=("checkpoint", "nunique"),
        )
        .sort_values(["condition", "layer"])
        .reset_index(drop=True)
    )

    for col in ["spearman_std", "pearson_std"]:
        summary[col] = summary[col].fillna(0.0)

    return summary


def build_stsb_delta_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    wanted = [
        "pretrained_frozen",
        "mean_finetuned",
        "flag_finetuned",
    ]

    pieces = {}
    for condition in wanted:
        sub = summary[
            summary["condition"] == condition
        ][
            [
                "layer",
                "layer_name",
                "spearman_mean",
                "pearson_mean",
            ]
        ].copy()

        sub = sub.rename(
            columns={
                "spearman_mean":
                    f"{condition}_spearman",
                "pearson_mean":
                    f"{condition}_pearson",
            }
        )

        pieces[condition] = sub

    merged = pieces[wanted[0]]
    for condition in wanted[1:]:
        merged = merged.merge(
            pieces[condition],
            on=["layer", "layer_name"],
            how="outer",
        )

    merged = merged.sort_values("layer").reset_index(drop=True)

    merged["mean_minus_frozen_spearman"] = (
        merged["mean_finetuned_spearman"]
        - merged["pretrained_frozen_spearman"]
    )
    merged["flag_minus_frozen_spearman"] = (
        merged["flag_finetuned_spearman"]
        - merged["pretrained_frozen_spearman"]
    )
    merged["mean_minus_flag_spearman"] = (
        merged["mean_finetuned_spearman"]
        - merged["flag_finetuned_spearman"]
    )

    return merged


def plot_controls(
    stsb_summary: pd.DataFrame,
    sprint_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    conditions = [
        (
            "pretrained_frozen",
            "STS-B on pretrained/frozen RoBERTa",
        ),
        (
            "mean_finetuned",
            "Mean-trained STS-B backbone",
        ),
        (
            "flag_finetuned",
            "FLaG-trained STS-B backbone",
        ),
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 4.8),
    )

    ax = axes[0]

    for condition, label in conditions:
        sub = (
            stsb_summary[
                stsb_summary["condition"] == condition
            ]
            .sort_values("layer")
        )

        x = sub["layer"].to_numpy()
        y = sub["spearman_mean"].to_numpy()
        std = sub["spearman_std"].to_numpy()

        ax.plot(
            x,
            y,
            marker="o",
            label=label,
        )

        if np.any(std > 0):
            ax.fill_between(
                x,
                y - std,
                y + std,
                alpha=0.12,
            )

    max_layer = int(stsb_summary["layer"].max())
    ticks = np.arange(max_layer + 1)

    ax.set_title(
        "STS-B: where DC-readable task information emerges"
    )
    ax.set_xlabel("RoBERTa representation")
    ax.set_ylabel("Spearman")
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        ["Emb"] + [str(i) for i in range(1, max_layer + 1)]
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]

    sprint_sorted = sprint_df.sort_values("layer")
    x = sprint_sorted["layer"].to_numpy()
    y = sprint_sorted["average_precision"].to_numpy()

    ax.plot(
        x,
        y,
        marker="o",
        label="Sprint, pretrained/frozen RoBERTa",
    )
    ax.set_title(
        "Sprint: DC-readable matching signal"
    )
    ax.set_xlabel("RoBERTa representation")
    ax.set_ylabel("Average Precision")
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        ["Emb"] + [str(i) for i in range(1, max_layer + 1)]
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Layer-wise DC/global-mean task information with backbone controls",
        fontsize=13,
    )
    fig.tight_layout()

    fig.savefig(
        output_dir / "dc_layer_controls.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "dc_layer_controls.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_stsb_deltas(
    delta_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    x = delta_df["layer"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.axhline(
        0.0,
        linewidth=1.0,
        alpha=0.5,
    )

    ax.plot(
        x,
        delta_df["mean_minus_frozen_spearman"],
        marker="o",
        label="Mean fine-tuning - frozen",
    )
    ax.plot(
        x,
        delta_df["flag_minus_frozen_spearman"],
        marker="o",
        label="FLaG fine-tuning - frozen",
    )
    ax.plot(
        x,
        delta_df["mean_minus_flag_spearman"],
        marker="o",
        label="Mean - FLaG",
    )

    max_layer = int(delta_df["layer"].max())
    ax.set_xticks(np.arange(max_layer + 1))
    ax.set_xticklabels(
        ["Emb"] + [str(i) for i in range(1, max_layer + 1)]
    )
    ax.set_xlabel("RoBERTa representation")
    ax.set_ylabel("Delta Spearman")
    ax.set_title(
        "STS-B: change in DC-readable task information"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(
        output_dir / "stsb_dc_layer_deltas.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "stsb_dc_layer_deltas.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Layer-wise DC controls: frozen vs Mean-finetuned vs "
            "FLaG-finetuned STS-B backbones, plus frozen Sprint."
        )
    )

    parser.add_argument(
        "--stsb_mean_checkpoints",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--stsb_flag_checkpoints",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path(
            "/home/data/home/wwr_lumos/models/roberta-base"
        ),
    )
    parser.add_argument(
        "--stsb_data_path",
        type=Path,
        default=Path(
            "/home/data/home/wwr_lumos/AMPCliff/"
            "data/text/stsbenchmark"
        ),
    )
    parser.add_argument(
        "--sprint_data_path",
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
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "/home/data/home/wwr_lumos/AMPCliff/"
            "outputs/text/dc_layer_controls"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    seed_everything(args.seed)

    for path in (
        list(args.stsb_mean_checkpoints)
        + list(args.stsb_flag_checkpoints)
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("device:", device)
    print("split:", args.split)
    print(
        "Mean checkpoints:",
        len(args.stsb_mean_checkpoints),
    )
    print(
        "FLaG checkpoints:",
        len(args.stsb_flag_checkpoints),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        local_files_only=True,
    )

    stsb_loader, stsb_n = make_loader(
        args.stsb_data_path,
        args.split,
        target_key="score",
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    sprint_loader, sprint_n = make_loader(
        args.sprint_data_path,
        args.split,
        target_key="label",
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    print(f"STS-B {args.split}: {stsb_n} pairs")
    print(f"Sprint {args.split}: {sprint_n} pairs")

    frozen_df = probe_stsb_frozen(
        model_path=args.model_path,
        loader=stsb_loader,
        device=device,
    )

    mean_df, mean_meta = probe_checkpoint_group(
        condition="mean_finetuned",
        model_path=args.model_path,
        checkpoints=list(args.stsb_mean_checkpoints),
        loader=stsb_loader,
        device=device,
    )

    flag_df, flag_meta = probe_checkpoint_group(
        condition="flag_finetuned",
        model_path=args.model_path,
        checkpoints=list(args.stsb_flag_checkpoints),
        loader=stsb_loader,
        device=device,
    )

    stsb_raw = pd.concat(
        [frozen_df, mean_df, flag_df],
        ignore_index=True,
    )

    stsb_summary = summarize_stsb_controls(stsb_raw)
    delta_df = build_stsb_delta_table(stsb_summary)

    sprint_df = probe_sprint(
        model_path=args.model_path,
        loader=sprint_loader,
        device=device,
    )

    stsb_raw.to_csv(
        args.output_dir / "stsb_dc_layer_controls_raw.csv",
        index=False,
    )
    stsb_summary.to_csv(
        args.output_dir / "stsb_dc_layer_controls_summary.csv",
        index=False,
    )
    delta_df.to_csv(
        args.output_dir / "stsb_dc_layer_control_deltas.csv",
        index=False,
    )
    sprint_df.to_csv(
        args.output_dir / "sprint_dc_layer_frozen.csv",
        index=False,
    )

    plot_controls(
        stsb_summary=stsb_summary,
        sprint_df=sprint_df,
        output_dir=args.output_dir,
    )
    plot_stsb_deltas(
        delta_df=delta_df,
        output_dir=args.output_dir,
    )

    metadata = {
        "probe": "layer-wise DC/global-mean controls",
        "split": args.split,
        "stsb_num_pairs": stsb_n,
        "sprint_num_pairs": sprint_n,
        "conditions": {
            "stsb_pretrained_frozen": (
                "pretrained RoBERTa-base, no task fine-tuning"
            ),
            "stsb_mean_finetuned": (
                "backbone weights from STS-B Mean-pooling checkpoints"
            ),
            "stsb_flag_finetuned": (
                "backbone weights from STS-B FLaG checkpoints; "
                "probe still reads only masked mean/DC at each layer"
            ),
            "sprint_pretrained_frozen": (
                "protocol-specified frozen pretrained RoBERTa-base"
            ),
        },
        "dc_definition": (
            "masked mean over all valid tokens including special tokens"
        ),
        "stsb_mean_checkpoints": mean_meta,
        "stsb_flag_checkpoints": flag_meta,
    }

    with open(
        args.output_dir / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nFinal-layer comparison:")
    final = (
        stsb_summary[
            stsb_summary["layer"]
            == stsb_summary["layer"].max()
        ][
            [
                "condition",
                "spearman_mean",
                "spearman_std",
                "pearson_mean",
                "pearson_std",
            ]
        ]
    )
    print(final.to_string(index=False))

    print("\nSaved to:", args.output_dir)


if __name__ == "__main__":
    main()
