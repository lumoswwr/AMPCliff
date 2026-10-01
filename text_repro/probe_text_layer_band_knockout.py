#!/usr/bin/env python3
"""
Text layer x frequency-band knockout for FLaG.

Main mechanism question:
  At each RoBERTa transformer layer, how much does the final FLaG prediction
  degrade when one token-sequence DCT band is removed?

This mirrors the AMP sequence-frequency knockout logic:
  layer output -> DCT along valid token axis -> notch one band -> IDCT ->
  continue through later backbone layers -> FLaG -> final metric.

Important:
- Knockout is an evaluation-time intervention; no parameters are trained here.
- STS-B loads its task-finetuned FLaG checkpoint.
- Sprint loads its published-protocol FLaG checkpoint whose RoBERTa backbone
  remained frozen during training.
- preserve_norm=True by default to match the AMP knockout implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from spectrual_filter.filter_seq import (
    allocate_prism_bands,
    dct_ortho,
    idct_ortho,
)
from train_sts import (
    STSCollator,
    STSDataset,
    SentenceEncoder,
    correlation_metrics,
)
from train_sprint import (
    SprintCollator,
    SprintDataset,
    SprintPairClassifier,
)


def load_config(checkpoint: Path) -> Dict[str, object]:
    path = checkpoint.parent / "config.json"
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_state(model: torch.nn.Module, checkpoint: Path) -> Dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=True)
    return {
        "epoch": payload.get("epoch"),
        "val_spearman": payload.get("val_spearman"),
        "val_accuracy": payload.get("val_accuracy"),
        "val_average_precision": payload.get("val_average_precision"),
    }


def text_band_mask(
    length: int,
    band: int,
    *,
    k_bands: int,
    base: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Match the AMP Prism allocation whenever length >= k_bands.

    Very short text sequences can have length < 8, unlike the AMP peptide
    setup. For those only, use a deterministic relative-frequency fallback:
    coefficient j maps to floor(j * k_bands / length). This keeps DC in B0
    and permits empty high/intermediate bands rather than dropping samples.
    """
    if length >= k_bands:
        masks, _, _, _ = allocate_prism_bands(
            length,
            k=k_bands,
            base=base,
        )
        return masks[band].to(device=device, dtype=dtype)

    idx = torch.arange(length, device=device)
    assigned = torch.div(
        idx * k_bands,
        length,
        rounding_mode="floor",
    )
    return (assigned == band).to(dtype=dtype)


def notch_band_grouped(
    hidden: torch.Tensor,
    lengths: torch.Tensor,
    *,
    band: int,
    k_bands: int,
    base: int,
    preserve_norm: bool,
) -> torch.Tensor:
    assert hidden.dim() == 3
    out = hidden.clone()
    lengths = lengths.to(device=hidden.device, dtype=torch.long)

    for length_tensor in torch.unique(lengths):
        length = int(length_tensor.item())
        if length <= 1:
            continue

        rows = torch.nonzero(
            lengths == length,
            as_tuple=False,
        ).squeeze(-1)

        x = hidden.index_select(0, rows)[:, :length, :]
        coeff = dct_ortho(x, dim=1)

        mask = text_band_mask(
            length,
            band,
            k_bands=k_bands,
            base=base,
            device=hidden.device,
            dtype=hidden.dtype,
        ).view(1, length, 1)

        filtered = coeff * (1.0 - mask)
        recon = idct_ortho(filtered, dim=1)

        if preserve_norm:
            eps = 1e-8
            original_norm = x.norm(
                p=2,
                dim=-1,
                keepdim=True,
            )
            recon_norm = recon.norm(
                p=2,
                dim=-1,
                keepdim=True,
            )
            recon = recon * (
                (original_norm + eps)
                / (recon_norm + eps)
            )

        selected = out.index_select(0, rows)
        selected[:, :length, :] = recon
        out.index_copy_(0, rows, selected)

    return out


class LayerBandKnockout:
    def __init__(
        self,
        layer: torch.nn.Module,
        *,
        band: int,
        k_bands: int,
        base: int,
        preserve_norm: bool,
    ):
        self.layer = layer
        self.band = int(band)
        self.k_bands = int(k_bands)
        self.base = int(base)
        self.preserve_norm = bool(preserve_norm)
        self.lengths = None
        self.handle = None

    def set_attention_masks(
        self,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
    ) -> None:
        combined = torch.cat([mask1, mask2], dim=0)
        self.lengths = combined.long().sum(dim=1)

    def _hook(self, module, inputs, output):
        if self.lengths is None:
            raise RuntimeError(
                "Knockout lengths were not set before backbone forward."
            )

        if isinstance(output, tuple):
            hidden = output[0]
            new_hidden = notch_band_grouped(
                hidden,
                self.lengths,
                band=self.band,
                k_bands=self.k_bands,
                base=self.base,
                preserve_norm=self.preserve_norm,
            )
            return (new_hidden,) + output[1:]

        if isinstance(output, list):
            hidden = output[0]
            new_hidden = notch_band_grouped(
                hidden,
                self.lengths,
                band=self.band,
                k_bands=self.k_bands,
                base=self.base,
                preserve_norm=self.preserve_norm,
            )
            return [new_hidden] + output[1:]

        if torch.is_tensor(output):
            return notch_band_grouped(
                output,
                self.lengths,
                band=self.band,
                k_bands=self.k_bands,
                base=self.base,
                preserve_norm=self.preserve_norm,
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


@torch.no_grad()
def eval_stsb(
    model: SentenceEncoder,
    loader: DataLoader,
    device: torch.device,
    knockout: LayerBandKnockout | None = None,
) -> Dict[str, float]:
    model.eval()
    preds: List[float] = []
    labels: List[float] = []

    for tok1, tok2, score in loader:
        tok1 = {
            k: v.to(device)
            for k, v in tok1.items()
        }
        tok2 = {
            k: v.to(device)
            for k, v in tok2.items()
        }

        if knockout is not None:
            knockout.set_attention_masks(
                tok1["attention_mask"],
                tok2["attention_mask"],
            )

        z1, z2 = model(tok1, tok2)
        sim = F.cosine_similarity(z1, z2, dim=-1)

        preds.extend(
            sim.detach().cpu().numpy().tolist()
        )
        labels.extend(score.numpy().tolist())

    return correlation_metrics(preds, labels)


@torch.no_grad()
def eval_sprint(
    model: SprintPairClassifier,
    loader: DataLoader,
    device: torch.device,
    knockout: LayerBandKnockout | None = None,
) -> Dict[str, float]:
    model.eval()
    cosines: List[float] = []
    labels: List[int] = []

    for tok1, tok2, y in loader:
        tok1 = {
            k: v.to(device)
            for k, v in tok1.items()
        }
        tok2 = {
            k: v.to(device)
            for k, v in tok2.items()
        }

        if knockout is not None:
            knockout.set_attention_masks(
                tok1["attention_mask"],
                tok2["attention_mask"],
            )

        _, cosine = model(tok1, tok2)

        cosines.extend(
            cosine.detach().cpu().numpy().tolist()
        )
        labels.extend(
            y.numpy().astype(np.int64).tolist()
        )

    ap = average_precision_score(
        np.asarray(labels, dtype=np.int64),
        np.asarray(cosines, dtype=np.float64),
    )
    return {
        "average_precision": float(ap),
    }


def build_stsb(
    model_path: Path,
    checkpoint: Path,
    device: torch.device,
):
    config = load_config(checkpoint)
    pooling = str(config.get("pooling", "FLaG"))
    if pooling != "FLaG":
        raise ValueError(
            f"STS-B checkpoint must be FLaG, got pooling={pooling}"
        )

    model = SentenceEncoder(
        model_path=str(model_path),
        pooling="FLaG",
        fixed_fft_length=config.get("fixed_fft_length"),
        pool_dropout=float(config.get("pool_dropout", 0.0)),
        post_pool_norm=bool(int(config.get("post_pool_norm", 1))),
    ).to(device)

    meta = load_state(model, checkpoint)
    return model, config, meta


def build_sprint(
    model_path: Path,
    checkpoint: Path,
    device: torch.device,
):
    config = load_config(checkpoint)
    pooling = str(config.get("pooling", "FLaG"))
    if pooling != "FLaG":
        raise ValueError(
            f"Sprint checkpoint must be FLaG, got pooling={pooling}"
        )

    if bool(config.get("finetune_backbone", False)):
        raise ValueError(
            "Main Sprint knockout expects the published frozen-backbone "
            "FLaG checkpoint, not an unfrozen control."
        )

    model = SprintPairClassifier(
        model_path=str(model_path),
        pooling="FLaG",
        finetune_backbone=False,
    ).to(device)

    meta = load_state(model, checkpoint)
    return model, config, meta


def make_stsb_loader(
    tokenizer,
    data_path: Path,
    split: str,
    batch_size: int,
    max_length: int,
):
    raw = load_from_disk(str(data_path))
    ds = STSDataset(raw[split])
    collator = STSCollator(tokenizer, max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    ), len(ds)


def make_sprint_loader(
    tokenizer,
    data_path: Path,
    split: str,
    batch_size: int,
    max_length: int,
):
    raw = load_from_disk(str(data_path))
    ds = SprintDataset(raw[split])
    collator = SprintCollator(tokenizer, max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    ), len(ds)


def run_grid(
    *,
    dataset: str,
    model,
    loader,
    device: torch.device,
    k_bands: int,
    base: int,
    preserve_norm: bool,
) -> pd.DataFrame:
    layers = model.backbone.encoder.layer
    num_layers = len(layers)

    if dataset == "stsb":
        baseline_metrics = eval_stsb(
            model,
            loader,
            device,
            knockout=None,
        )
        baseline = baseline_metrics["spearman"]
        metric_name = "spearman"
        evaluator = eval_stsb
    elif dataset == "sprint":
        baseline_metrics = eval_sprint(
            model,
            loader,
            device,
            knockout=None,
        )
        baseline = baseline_metrics["average_precision"]
        metric_name = "average_precision"
        evaluator = eval_sprint
    else:
        raise ValueError(dataset)

    print(
        f"[{dataset}] baseline {metric_name}={baseline:.6f}"
    )

    rows = []

    for layer_idx in range(num_layers):
        for band in range(k_bands):
            hook = LayerBandKnockout(
                layers[layer_idx],
                band=band,
                k_bands=k_bands,
                base=base,
                preserve_norm=preserve_norm,
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
            delta = baseline - value

            row = {
                "dataset": dataset,
                "layer": layer_idx + 1,
                "band": band,
                "baseline_metric": baseline,
                "knockout_metric": value,
                "delta_metric": delta,
                "metric_name": metric_name,
            }
            rows.append(row)

            print(
                f"[{dataset}] L{layer_idx + 1:02d} "
                f"B{band} {metric_name}={value:.6f} "
                f"delta={delta:+.6f}"
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
        values="delta_metric",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(8.0, 6.3))
    image = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(
        [f"B{x}" for x in pivot.columns]
    )
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(
        [f"L{x}" for x in pivot.index]
    )
    ax.set_xlabel("DCT frequency band removed")
    ax.set_ylabel("RoBERTa transformer layer")

    metric = frame["metric_name"].iloc[0]
    label = (
        "Spearman"
        if metric == "spearman"
        else "Average Precision"
    )
    ax.set_title(
        f"{dataset.upper()}: performance drop after layer-band knockout"
    )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(
        f"Baseline - knockout {label}"
    )

    fig.tight_layout()
    fig.savefig(
        output.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches="tight",
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
        "--stsb_split",
        choices=["validation", "test"],
        default="test",
    )
    parser.add_argument(
        "--sprint_split",
        choices=["validation", "test"],
        default="validation",
        help=(
            "Validation is the practical first-pass mechanism split; "
            "Sprint test is ~10x larger and can be run after sanity checks."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--k_bands", type=int, default=8)
    parser.add_argument("--base", type=int, default=4)
    parser.add_argument(
        "--preserve_norm",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "/home/data/home/wwr_lumos/AMPCliff/"
            "outputs/text/layer_band_knockout"
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

    print("device:", device)
    print("k_bands:", args.k_bands)
    print("base:", args.base)
    print("preserve_norm:", bool(args.preserve_norm))
    print("STS-B split:", args.stsb_split)
    print("Sprint split:", args.sprint_split)

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
        preserve_norm=bool(args.preserve_norm),
    )
    stsb_df.to_csv(
        args.output_dir / "stsb_layer_band_knockout.csv",
        index=False,
    )
    plot_heatmap(
        stsb_df,
        dataset="stsb",
        output=args.output_dir / "stsb_layer_band_knockout",
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
        preserve_norm=bool(args.preserve_norm),
    )
    sprint_df.to_csv(
        args.output_dir / "sprint_layer_band_knockout.csv",
        index=False,
    )
    plot_heatmap(
        sprint_df,
        dataset="sprint",
        output=args.output_dir / "sprint_layer_band_knockout",
    )

    combined = pd.concat(
        [stsb_df, sprint_df],
        ignore_index=True,
    )
    combined.to_csv(
        args.output_dir / "layer_band_knockout_combined.csv",
        index=False,
    )

    metadata = {
        "method": (
            "RoBERTa transformer-layer output DCT band-notch knockout, "
            "followed by continuation through later layers and FLaG."
        ),
        "k_bands": args.k_bands,
        "base": args.base,
        "preserve_norm": bool(args.preserve_norm),
        "short_sequence_fallback": (
            "For valid token length < k_bands only, coefficient j is "
            "assigned to floor(j*k_bands/length); DC remains B0. "
            "For length >= k_bands, the original AMP Prism allocation "
            "is used exactly."
        ),
        "stsb": {
            "split": args.stsb_split,
            "n_pairs": stsb_n,
            "checkpoint": str(args.stsb_checkpoint),
            "checkpoint_meta": stsb_meta,
            "training_protocol": (
                "task-finetuned RoBERTa backbone + trained FLaG"
            ),
            "config": stsb_config,
        },
        "sprint": {
            "split": args.sprint_split,
            "n_pairs": sprint_n,
            "checkpoint": str(args.sprint_checkpoint),
            "checkpoint_meta": sprint_meta,
            "training_protocol": (
                "pretrained/frozen RoBERTa backbone + trained FLaG/head"
            ),
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
