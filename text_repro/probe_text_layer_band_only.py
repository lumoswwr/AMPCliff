#!/usr/bin/env python3
"""
Frozen-backbone layer x DCT band-only sufficiency probe.

At a chosen RoBERTa layer:
  hidden -> DCT along valid-token axis -> KEEP one band -> IDCT
         -> continue through later RoBERTa layers -> trained FLaG -> metric

This complements knockout:
  knockout asks necessity/dependency;
  band-only asks sufficiency.

For L12 there are no later transformer layers after the intervention, so L12
is the cleanest direct test of how much task performance FLaG can recover from
one band alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from probe_text_layer_band_knockout import (
    build_sprint,
    build_stsb,
    dct_self_check,
    dct_ortho,
    eval_sprint,
    eval_stsb,
    idct_ortho,
    make_sprint_loader,
    make_stsb_loader,
    text_band_mask,
)
from transformers import AutoTokenizer


def pass_band_grouped(
    hidden: torch.Tensor,
    lengths: torch.Tensor,
    *,
    band: int,
    k_bands: int,
    base: int,
) -> torch.Tensor:
    """Keep only one DCT band; do not restore token norms."""
    assert hidden.dim() == 3
    out = hidden.clone()
    lengths = lengths.to(
        device=hidden.device,
        dtype=torch.long,
    )

    for length_tensor in torch.unique(lengths):
        length = int(length_tensor.item())
        if length <= 1:
            continue

        rows = torch.nonzero(
            lengths == length,
            as_tuple=False,
        ).squeeze(-1)

        x = hidden.index_select(
            0,
            rows,
        )[:, :length, :]

        coeff = dct_ortho(x, dim=1)
        mask = text_band_mask(
            length,
            band,
            k_bands=k_bands,
            base=base,
            device=hidden.device,
            dtype=hidden.dtype,
        ).view(1, length, 1)

        filtered = coeff * mask
        recon = idct_ortho(
            filtered,
            dim=1,
        )

        selected = out.index_select(0, rows)
        selected[:, :length, :] = recon
        out.index_copy_(0, rows, selected)

    return out


class LayerBandPass:
    def __init__(
        self,
        layer: torch.nn.Module,
        *,
        band: int,
        k_bands: int,
        base: int,
    ):
        self.layer = layer
        self.band = int(band)
        self.k_bands = int(k_bands)
        self.base = int(base)
        self.lengths = None
        self.handle = None

    def set_attention_masks(
        self,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
    ) -> None:
        combined = torch.cat(
            [mask1, mask2],
            dim=0,
        )
        self.lengths = combined.long().sum(dim=1)

    def _hook(self, module, inputs, output):
        if self.lengths is None:
            raise RuntimeError(
                "Band-pass lengths were not set before backbone forward."
            )

        if isinstance(output, tuple):
            hidden = output[0]
            passed = pass_band_grouped(
                hidden,
                self.lengths,
                band=self.band,
                k_bands=self.k_bands,
                base=self.base,
            )
            return (passed,) + output[1:]

        if isinstance(output, list):
            hidden = output[0]
            passed = pass_band_grouped(
                hidden,
                self.lengths,
                band=self.band,
                k_bands=self.k_bands,
                base=self.base,
            )
            return [passed] + output[1:]

        if torch.is_tensor(output):
            return pass_band_grouped(
                output,
                self.lengths,
                band=self.band,
                k_bands=self.k_bands,
                base=self.base,
            )

        raise TypeError(
            f"Unsupported transformer-layer output type: {type(output)}"
        )

    def register(self) -> None:
        self.handle = self.layer.register_forward_hook(
            self._hook
        )

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def run_grid(
    *,
    dataset: str,
    model,
    loader,
    device: torch.device,
    k_bands: int,
    base: int,
) -> pd.DataFrame:
    layers = model.backbone.encoder.layer

    if dataset == "stsb":
        evaluator = eval_stsb
        metric_name = "spearman"
    elif dataset == "sprint":
        evaluator = eval_sprint
        metric_name = "average_precision"
    else:
        raise ValueError(dataset)

    baseline_metrics = evaluator(
        model,
        loader,
        device,
        knockout=None,
    )
    baseline = baseline_metrics[metric_name]

    print(
        f"[{dataset}] baseline {metric_name}={baseline:.6f}"
    )

    if dataset == "sprint" and baseline < 0.20:
        raise RuntimeError(
            "Sprint baseline AP is implausibly low. "
            "Use the audited sprint_frozen_flag checkpoint."
        )

    rows = []

    for layer_idx, layer in enumerate(layers):
        for band in range(k_bands):
            hook = LayerBandPass(
                layer,
                band=band,
                k_bands=k_bands,
                base=base,
            )
            hook.register()

            try:
                metrics = evaluator(
                    model,
                    loader,
                    device,
                    knockout=hook,
                )
            finally:
                hook.remove()

            value = metrics[metric_name]
            fraction = (
                value / baseline
                if abs(baseline) > 1e-12
                else float("nan")
            )

            rows.append({
                "dataset": dataset,
                "layer": layer_idx + 1,
                "band": band,
                "baseline_metric": baseline,
                "band_only_metric": value,
                "fraction_of_baseline": fraction,
                "metric_name": metric_name,
            })

            print(
                f"[{dataset}] L{layer_idx + 1:02d} B{band} "
                f"{metric_name}={value:.6f} "
                f"fraction={fraction:.3f}"
            )

    return pd.DataFrame(rows)


def plot_heatmap(
    frame: pd.DataFrame,
    *,
    dataset: str,
    output: Path,
) -> None:
    pivot = frame.pivot(
        index="layer",
        columns="band",
        values="band_only_metric",
    ).sort_index()

    values = pivot.to_numpy(dtype=float)
    vmax = float(np.quantile(values, 0.99))
    vmin = float(np.nanmin(values))
    if vmax <= vmin:
        vmax = float(np.nanmax(values))
    if vmax <= vmin:
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    image = ax.imshow(
        values,
        cmap="GnBu",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
        origin="upper",
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [rf"$\mathcal{{B}}_{{{int(x)}}}$" for x in pivot.columns]
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(
        [f"L{int(x)}" for x in pivot.index]
    )

    ax.set_xlabel("Sequence-frequency band retained")
    ax.set_ylabel("RoBERTa transformer layer")
    ax.set_title(
        f"{dataset.upper()}: single-band sufficiency"
    )

    for x in np.arange(-0.5, len(pivot.columns), 1):
        ax.axvline(
            x,
            linewidth=0.45,
            color="white",
            alpha=0.75,
        )
    for y in np.arange(-0.5, len(pivot.index), 1):
        ax.axhline(
            y,
            linewidth=0.45,
            color="white",
            alpha=0.75,
        )

    metric = frame["metric_name"].iloc[0]
    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.045,
        pad=0.035,
        extend="max",
    )
    cbar.set_label(
        "Spearman"
        if metric == "spearman"
        else "Average Precision"
    )

    fig.tight_layout()
    fig.savefig(
        output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stsb_checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--sprint_checkpoint",
        type=Path,
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
            "/home/data/home/wwr_lumos/AMPCliff/data/text/stsbenchmark"
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
        "--stsb_split",
        choices=["validation", "test"],
        default="test",
    )
    parser.add_argument(
        "--sprint_split",
        choices=["validation", "test"],
        default="validation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--k_bands",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--base",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "outputs/text/layer_band_only_frozen_seed0"
        ),
    )

    args = parser.parse_args()

    for p in [
        args.stsb_checkpoint,
        args.sprint_checkpoint,
    ]:
        if not p.is_file():
            raise FileNotFoundError(p)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dct_check = dct_self_check(device)
    print("device:", device)
    print("DCT self-check:", dct_check)
    print(
        "Band-only mode: keep one band; "
        "NO norm restoration (AMP pass-mode compatible)."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        local_files_only=True,
    )

    stsb_loader, stsb_n = make_stsb_loader(
        tokenizer,
        args.stsb_data_path,
        args.stsb_split,
        args.batch_size,
        args.max_length,
    )
    sprint_loader, sprint_n = make_sprint_loader(
        tokenizer,
        args.sprint_data_path,
        args.sprint_split,
        args.batch_size,
        args.max_length,
    )

    stsb_model, stsb_config, stsb_meta = build_stsb(
        args.model_path,
        args.stsb_checkpoint,
        device,
    )
    stsb_df = run_grid(
        dataset="stsb",
        model=stsb_model,
        loader=stsb_loader,
        device=device,
        k_bands=args.k_bands,
        base=args.base,
    )
    stsb_df.to_csv(
        args.output_dir / "stsb_layer_band_only.csv",
        index=False,
    )
    plot_heatmap(
        stsb_df,
        dataset="stsb",
        output=args.output_dir / "stsb_layer_band_only",
    )

    del stsb_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sprint_model, sprint_config, sprint_meta = build_sprint(
        args.model_path,
        args.sprint_checkpoint,
        device,
    )
    sprint_df = run_grid(
        dataset="sprint",
        model=sprint_model,
        loader=sprint_loader,
        device=device,
        k_bands=args.k_bands,
        base=args.base,
    )
    sprint_df.to_csv(
        args.output_dir / "sprint_layer_band_only.csv",
        index=False,
    )
    plot_heatmap(
        sprint_df,
        dataset="sprint",
        output=args.output_dir / "sprint_layer_band_only",
    )

    pd.concat(
        [stsb_df, sprint_df],
        ignore_index=True,
    ).to_csv(
        args.output_dir / "layer_band_only_combined.csv",
        index=False,
    )

    metadata = {
        "method": (
            "Frozen-backbone RoBERTa layer output DCT single-band pass, "
            "IDCT, continuation through later layers, then trained FLaG."
        ),
        "interpretation": (
            "Band-only is a sufficiency probe. L12 is the cleanest direct "
            "single-band test because no later transformer layer remains."
        ),
        "k_bands": args.k_bands,
        "base": args.base,
        "norm_restoration": False,
        "dct_self_check": dct_check,
        "stsb": {
            "split": args.stsb_split,
            "n_pairs": stsb_n,
            "checkpoint": str(args.stsb_checkpoint),
            "checkpoint_meta": stsb_meta,
            "config": stsb_config,
        },
        "sprint": {
            "split": args.sprint_split,
            "n_pairs": sprint_n,
            "checkpoint": str(args.sprint_checkpoint),
            "checkpoint_meta": sprint_meta,
            "config": sprint_config,
        },
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

    print("\nSaved to:", args.output_dir)


if __name__ == "__main__":
    main()
