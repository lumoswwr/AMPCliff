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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Keep this probe self-contained instead of importing
# spectrual_filter/filter_seq.py. The original AMP helper imports torch_dct
# unconditionally, but torch_dct is not listed in this repository's
# environment.yaml / requirements snapshot. We implement the same orthonormal
# DCT-II / inverse transform directly in PyTorch below.
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



_DCT_MATRIX_CACHE: Dict[Tuple[int, str, int, torch.dtype], torch.Tensor] = {}


def _device_key(device: torch.device) -> Tuple[str, int]:
    return (
        device.type,
        -1 if device.index is None else int(device.index),
    )


def dct_matrix_ortho(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Orthonormal DCT-II matrix C with:
        X = C @ x
        x = C.T @ X

    C[0, :] = 1 / sqrt(n), so the first coefficient is exactly
    sqrt(n) * mean(x), i.e. the DC/global-mean component up to scale.
    """
    dev_type, dev_index = _device_key(device)
    key = (int(n), dev_type, dev_index, dtype)

    cached = _DCT_MATRIX_CACHE.get(key)
    if cached is not None:
        return cached

    # Build in float64 for accurate basis construction, then cast.
    k = torch.arange(
        n,
        device=device,
        dtype=torch.float64,
    ).unsqueeze(1)
    t = torch.arange(
        n,
        device=device,
        dtype=torch.float64,
    ).unsqueeze(0)

    basis = torch.cos(
        torch.pi
        / float(n)
        * (t + 0.5)
        * k
    )

    scale = torch.full(
        (n, 1),
        (2.0 / float(n)) ** 0.5,
        device=device,
        dtype=torch.float64,
    )
    scale[0, 0] = (1.0 / float(n)) ** 0.5

    matrix = (basis * scale).to(dtype=dtype)
    _DCT_MATRIX_CACHE[key] = matrix
    return matrix


def dct_ortho(
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    if dim < 0:
        dim += x.dim()

    moved = x.movedim(dim, -1)
    n = moved.size(-1)
    matrix = dct_matrix_ortho(
        n,
        device=x.device,
        dtype=x.dtype,
    )

    # (..., n) @ C.T -> (..., n)
    transformed = torch.matmul(
        moved,
        matrix.transpose(0, 1),
    )
    return transformed.movedim(-1, dim)


def idct_ortho(
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    if dim < 0:
        dim += x.dim()

    moved = x.movedim(dim, -1)
    n = moved.size(-1)
    matrix = dct_matrix_ortho(
        n,
        device=x.device,
        dtype=x.dtype,
    )

    # Orthonormal inverse: (..., n) @ C
    reconstructed = torch.matmul(
        moved,
        matrix,
    )
    return reconstructed.movedim(-1, dim)


def allocate_prism_bands(
    n: int,
    k: int = 5,
    base: int = 4,
):
    """
    Exact copy of the AMP Prism band-allocation rule, kept local so this
    analysis does not depend on torch_dct.
    """
    assert n >= k >= 1

    sizes = torch.ones(k, dtype=torch.long)
    remaining = n - k

    i = torch.arange(k, dtype=torch.float32)
    weights = base ** i
    frac = remaining * (weights / weights.sum())
    floor = torch.floor(frac).to(torch.long)
    sizes += floor

    left = int(remaining - int(floor.sum()))
    residual = (frac - floor.float()).tolist()
    order = sorted(
        range(k),
        key=lambda idx: residual[idx],
        reverse=True,
    )

    for j in range(left):
        sizes[order[j]] += 1

    ends = torch.cumsum(sizes, dim=0)
    starts = torch.cat(
        [
            torch.tensor([0], dtype=torch.long),
            ends[:-1],
        ]
    )

    masks = []
    for start, end in zip(
        starts.tolist(),
        ends.tolist(),
    ):
        mask = torch.zeros(n)
        mask[start:end] = 1.0
        masks.append(mask)

    return masks, sizes, starts, ends


def dct_self_check(device: torch.device) -> Dict[str, float]:
    """
    Validate the local DCT before running the expensive 12x8 grid:
    1) DCT -> IDCT reconstructs the input.
    2) coefficient 0 equals sqrt(T) * token mean.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)

    x = torch.randn(
        3,
        17,
        11,
        generator=generator,
        dtype=torch.float32,
    ).to(device)

    coeff = dct_ortho(x, dim=1)
    recon = idct_ortho(coeff, dim=1)

    recon_error = float(
        (recon - x).abs().max().detach().cpu()
    )

    expected_dc = (
        x.mean(dim=1)
        * (x.size(1) ** 0.5)
    )
    dc_error = float(
        (
            coeff[:, 0, :]
            - expected_dc
        )
        .abs()
        .max()
        .detach()
        .cpu()
    )

    if recon_error > 1e-4 or dc_error > 1e-4:
        raise RuntimeError(
            "Local orthonormal DCT self-check failed: "
            f"reconstruction_error={recon_error:.3e}, "
            f"dc_error={dc_error:.3e}"
        )

    return {
        "max_reconstruction_error": recon_error,
        "max_dc_equivalence_error": dc_error,
    }


def load_config(checkpoint: Path) -> Dict[str, object]:
    path = checkpoint.parent / "config.json"
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_meta(checkpoint: Path) -> Dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu")
    return {
        "epoch": payload.get("epoch"),
        "val_spearman": payload.get("val_spearman"),
        "val_accuracy": payload.get("val_accuracy"),
        "val_average_precision": payload.get("val_average_precision"),
    }


def load_state(model: torch.nn.Module, checkpoint: Path) -> Dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=True)
    return checkpoint_meta(checkpoint)


def load_sprint_state_for_cosine_probe(
    model: torch.nn.Module,
    checkpoint: Path,
) -> Dict[str, object]:
    """
    Main Sprint knockout must use the audited frozen-FLaG checkpoint format.

    Historical sprint_flag checkpoints used an older free Linear head and are
    not interchangeable with the canonical monotone-cosine reproduction.
    Refuse those checkpoints instead of silently bypassing the mismatch.
    """
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model_state_dict", payload)

    try:
        model.load_state_dict(
            state,
            strict=True,
        )
    except RuntimeError as exc:
        keys = set(state.keys())
        if (
            "head.linear.weight" in keys
            or "head.linear.bias" in keys
        ):
            raise RuntimeError(
                "This is a legacy Sprint checkpoint with head.linear.*. "
                "Do NOT use it for the main layer-band knockout. "
                "Use the audited canonical checkpoint under "
                "experiments/sprint_frozen_flag/seed_0/best_model.pt, "
                "whose validation cosine AP is ~0.758 for seed 0."
            ) from exc
        raise

    meta = checkpoint_meta(checkpoint)
    meta["head_used_for_metric"] = False
    meta["metric_path"] = "encoder -> cosine -> AP"
    return meta


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
    knockout: Optional[LayerBandKnockout] = None,
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
    knockout: Optional[LayerBandKnockout] = None,
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

        # Bypass the calibration head entirely. Sprint knockout is evaluated
        # with cosine AP, which is exactly the ranking quantity of interest
        # and is invariant to a positive affine calibration head.
        z1, z2 = model.encoder(tok1, tok2)
        cosine = F.cosine_similarity(
            z1,
            z2,
            dim=-1,
        )

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

    if not bool(config.get("freeze_backbone", False)):
        raise ValueError(
            "Frozen-only knockout report requires an STS-B checkpoint "
            "trained with --freeze_backbone."
        )

    model = SentenceEncoder(
        model_path=str(model_path),
        pooling="FLaG",
        fixed_fft_length=config.get("fixed_fft_length"),
        pool_dropout=float(config.get("pool_dropout", 0.0)),
        post_pool_norm=bool(int(config.get("post_pool_norm", 1))),
        freeze_backbone=True,
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

    meta = load_sprint_state_for_cosine_probe(
        model,
        checkpoint,
    )
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

        # The audited seed-0 frozen-FLaG reproduction has validation cosine
        # AP ~= 0.758. A near-random baseline (~0.01-0.03 on this 1% positive
        # task) signals a wrong/legacy checkpoint or architecture mismatch.
        if baseline < 0.20:
            raise RuntimeError(
                "Sprint baseline cosine AP is implausibly low "
                f"({baseline:.6f}). Stop before running the 12x8 grid. "
                "For the main experiment use "
                "experiments/sprint_frozen_flag/seed_0/best_model.pt."
            )
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
    parser.add_argument(
        "--skip_stsb",
        action="store_true",
        help=(
            "Reuse an already completed stsb_layer_band_knockout.csv "
            "from output_dir and run only Sprint."
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
    print(
        "DCT self-check:",
        f"recon_max={dct_check['max_reconstruction_error']:.3e}",
        f"dc_max={dct_check['max_dc_equivalence_error']:.3e}",
    )
    print("k_bands:", args.k_bands)
    print("base:", args.base)
    print("preserve_norm:", bool(args.preserve_norm))
    print("STS-B split:", args.stsb_split)
    print("Sprint split:", args.sprint_split)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        local_files_only=True,
    )

    sprint_loader, sprint_n = make_sprint_loader(
        tokenizer,
        args.sprint_data_path,
        args.sprint_split,
        args.batch_size,
        args.max_length,
    )

    stsb_csv = (
        args.output_dir
        / "stsb_layer_band_knockout.csv"
    )

    if args.skip_stsb:
        if not stsb_csv.is_file():
            raise FileNotFoundError(
                "--skip_stsb requested, but completed STS-B CSV "
                f"was not found: {stsb_csv}"
            )

        print(
            "[stsb] reusing completed results:",
            stsb_csv,
        )
        stsb_df = pd.read_csv(stsb_csv)
        stsb_config = load_config(args.stsb_checkpoint)
        stsb_meta = checkpoint_meta(args.stsb_checkpoint)
        stsb_n = int(
            load_from_disk(
                str(args.stsb_data_path)
            )[args.stsb_split].num_rows
        )
    else:
        stsb_loader, stsb_n = make_stsb_loader(
            tokenizer,
            args.stsb_data_path,
            args.stsb_split,
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
            stsb_csv,
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
        "dct_backend": (
            "self-contained orthonormal DCT-II matrix in PyTorch; "
            "avoids undeclared torch_dct dependency"
        ),
        "dct_self_check": dct_check,
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
                "pretrained/frozen RoBERTa backbone + trained FLaG"
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
