#!/usr/bin/env python3
"""Replot frozen text knockout heatmaps in the AMP paper visual style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def band_label(i: int) -> str:
    return rf"$\mathcal{{B}}_{{{int(i)}}}$"


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "layer",
        "band",
        "baseline_metric",
        "delta_metric",
        "metric_name",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} missing columns: {sorted(missing)}"
        )
    df = df.copy()
    df["abs_delta"] = df["delta_metric"].abs()
    df["relative_abs_drop"] = (
        df["abs_delta"]
        / df["baseline_metric"].abs().clip(lower=1e-12)
    )
    return df


def pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return (
        df.pivot(
            index="layer",
            columns="band",
            values=value,
        )
        .sort_index()
        .sort_index(axis=1)
    )


def paper_heatmap(
    ax,
    matrix: pd.DataFrame,
    *,
    title: str,
    norm: Normalize,
    cmap,
    ylabel: bool,
):
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
        origin="upper",
    )

    ax.set_xticks(
        np.arange(len(matrix.columns))
    )
    ax.set_xticklabels(
        [band_label(x) for x in matrix.columns],
        fontsize=12,
        fontweight="bold",
    )
    ax.set_yticks(
        np.arange(len(matrix.index))
    )
    ax.set_yticklabels(
        [f"L{int(x)}" for x in matrix.index],
        fontsize=11,
    )

    ax.set_xlabel(
        "Sequence-frequency band",
        fontsize=13,
    )
    ax.set_ylabel(
        "RoBERTa layer" if ylabel else "",
        fontsize=13,
    )
    ax.set_title(
        title,
        fontsize=15,
        pad=10,
    )

    # Thin white cell boundaries, matching the AMP Exp1 figure style.
    ax.set_xticks(
        np.arange(-0.5, len(matrix.columns), 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, len(matrix.index), 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        color="white",
        linewidth=0.55,
        alpha=0.85,
    )
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    return image


def task_figure(
    df: pd.DataFrame,
    *,
    task: str,
    out_base: Path,
) -> None:
    matrix = pivot(df, "abs_delta")
    values = matrix.to_numpy(dtype=float)
    vmax = float(
        np.quantile(values[np.isfinite(values)], 0.99)
    )
    if vmax <= 0:
        vmax = float(np.nanmax(values)) or 1.0

    fig, ax = plt.subplots(
        figsize=(8.0, 6.2)
    )
    image = paper_heatmap(
        ax,
        matrix,
        title=(
            f"{task}: layer × band knockout "
            r"$|\Delta \mathrm{performance}|$"
        ),
        norm=Normalize(vmin=0.0, vmax=vmax),
        cmap=plt.get_cmap("GnBu"),
        ylabel=True,
    )

    metric = str(df["metric_name"].iloc[0])
    metric_label = (
        "Spearman"
        if metric == "spearman"
        else "Average Precision"
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.035,
        extend="max",
    )
    cbar.set_label(
        rf"$|\Delta {metric_label}|$",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(
        out_base.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        out_base.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def combined_relative(
    stsb: pd.DataFrame,
    sprint: pd.DataFrame,
    *,
    out_base: Path,
) -> None:
    matrices = [
        pivot(stsb, "relative_abs_drop"),
        pivot(sprint, "relative_abs_drop"),
    ]

    all_values = np.concatenate(
        [
            m.to_numpy(dtype=float).ravel()
            for m in matrices
        ]
    )
    all_values = all_values[np.isfinite(all_values)]
    vmax = float(np.quantile(all_values, 0.99))
    vmax = max(vmax, 1e-6)
    norm = Normalize(
        vmin=0.0,
        vmax=vmax,
    )
    cmap = plt.get_cmap("GnBu")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 6.0),
        sharey=True,
    )

    image = paper_heatmap(
        axes[0],
        matrices[0],
        title="STSB",
        norm=norm,
        cmap=cmap,
        ylabel=True,
    )
    paper_heatmap(
        axes[1],
        matrices[1],
        title="SprintDuplicateQuestions",
        norm=norm,
        cmap=cmap,
        ylabel=False,
    )

    fig.suptitle(
        "Frozen RoBERTa + FLaG: relative sensitivity to sequence-frequency knockout",
        fontsize=16,
        y=1.01,
    )

    cbar = fig.colorbar(
        image,
        ax=axes,
        fraction=0.025,
        pad=0.025,
        extend="max",
    )
    cbar.set_label(
        r"$|\Delta M| / M_{baseline}$",
        fontsize=12,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.90,
        bottom=0.12,
        top=0.87,
        wspace=0.12,
    )
    fig.savefig(
        out_base.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        out_base.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(
            "outputs/text/layer_band_knockout_frozen_seed0"
        ),
    )
    args = parser.parse_args()

    stsb_path = (
        args.input_dir
        / "stsb_layer_band_knockout.csv"
    )
    sprint_path = (
        args.input_dir
        / "sprint_layer_band_knockout.csv"
    )

    stsb = load_table(stsb_path)
    sprint = load_table(sprint_path)

    task_figure(
        stsb,
        task="STS-B",
        out_base=(
            args.input_dir
            / "stsb_layer_band_knockout_paperstyle"
        ),
    )
    task_figure(
        sprint,
        task="Sprint",
        out_base=(
            args.input_dir
            / "sprint_layer_band_knockout_paperstyle"
        ),
    )
    combined_relative(
        stsb,
        sprint,
        out_base=(
            args.input_dir
            / "stsb_sprint_knockout_relative_paperstyle"
        ),
    )

    print(
        "Saved paper-style knockout figures to:",
        args.input_dir,
    )


if __name__ == "__main__":
    main()
