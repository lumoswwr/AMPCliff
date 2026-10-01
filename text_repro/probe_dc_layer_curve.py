#!/usr/bin/env python3
"""
Layer-wise DC-only probe for STS-B and SprintDuplicateQuestions.

Measure how much task-relevant information is readable from the DC / global
mean component of RoBERTa representations H0..H12.

STS-B:
  masked mean -> cosine similarity -> Spearman/Pearson.
  Use one or more Mean-pooling STS-B checkpoints so the backbone matches the
  trained Mean baseline.

Sprint:
  frozen pretrained RoBERTa -> masked mean -> cosine similarity -> AP.
  The trained Sprint cosine-logit head is a positive monotone transform, so
  Average Precision from cosine is identical to AP from that head.

The existing MeanPooling averages all valid tokens including RoBERTa special
tokens. This probe deliberately does the same so the final STS-B layer can
reproduce the Mean baseline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over all valid tokens, matching train_sts.py/train_sprint.py."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def infer_seed_from_path(path: Path, fallback: int) -> int:
    match = re.search(r"seed[_-](\d+)", str(path))
    return int(match.group(1)) if match else fallback


def correlation_metrics(pred: Sequence[float], gold: Sequence[float]) -> Dict[str, float]:
    pred_np = np.asarray(pred, dtype=np.float64)
    gold_np = np.asarray(gold, dtype=np.float64)
    return {
        "spearman": float(spearmanr(pred_np, gold_np).statistic),
        "pearson": float(pearsonr(pred_np, gold_np).statistic),
    }


def load_stsb_backbone_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> Dict[str, object]:
    """Load only RoBERTa backbone weights from a train_sts.py checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)

    backbone_state = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            backbone_state[key[len("backbone."):]] = value
        elif key.startswith("encoder.backbone."):
            backbone_state[key[len("encoder.backbone."):]] = value

    if not backbone_state:
        preview = list(state.keys())[:10]
        raise ValueError(
            f"No backbone weights found in {checkpoint_path}. "
            f"First keys: {preview}"
        )

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    bad_missing = [key for key in missing if not key.startswith("pooler.")]
    bad_unexpected = [key for key in unexpected if not key.startswith("pooler.")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Backbone checkpoint mismatch. "
            f"missing={bad_missing[:20]} unexpected={bad_unexpected[:20]}"
        )

    return {
        "epoch": checkpoint.get("epoch"),
        "val_spearman": checkpoint.get("val_spearman"),
    }


class PairDataset(Dataset):
    def __init__(self, split, target_key: str):
        self.split = split
        self.target_key = target_key

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int):
        row = self.split[idx]
        return {
            "sentence1": row["sentence1"],
            "sentence2": row["sentence2"],
            "target": float(row[self.target_key]),
        }


class PairCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        s1 = [row["sentence1"] for row in batch]
        s2 = [row["sentence2"] for row in batch]
        target = torch.tensor(
            [row["target"] for row in batch],
            dtype=torch.float32,
        )

        all_sentences = s1 + s2
        tokens = self.tokenizer(
            all_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_size = len(batch)
        tok1 = {key: value[:batch_size] for key, value in tokens.items()}
        tok2 = {key: value[batch_size:] for key, value in tokens.items()}
        return tok1, tok2, target


def make_loader(
    data_path: Path,
    split_name: str,
    target_key: str,
    tokenizer,
    max_length: int,
    batch_size: int,
) -> Tuple[DataLoader, int]:
    raw = load_from_disk(str(data_path))
    if split_name not in raw:
        raise KeyError(
            f"Split {split_name!r} not found in {data_path}. "
            f"Available: {list(raw.keys())}"
        )
    ds = PairDataset(raw[split_name], target_key=target_key)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=PairCollator(tokenizer, max_length=max_length),
    )
    return loader, len(ds)


@torch.no_grad()
def collect_layer_cosines(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[List[float]], List[float]]:
    model.eval()
    all_layer_cosines = None
    all_targets: List[float] = []

    for tok1, tok2, target in loader:
        tok1 = {key: value.to(device) for key, value in tok1.items()}
        tok2 = {key: value.to(device) for key, value in tok2.items()}

        combined = {
            key: torch.cat([tok1[key], tok2[key]], dim=0)
            for key in tok1
            if key in tok2
        }

        outputs = model(
            **combined,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Backbone did not return hidden_states.")

        if all_layer_cosines is None:
            all_layer_cosines = [[] for _ in hidden_states]

        if len(all_layer_cosines) != len(hidden_states):
            raise RuntimeError(
                "Number of hidden-state layers changed across batches: "
                f"{len(all_layer_cosines)} vs {len(hidden_states)}"
            )

        batch_size = tok1["input_ids"].size(0)
        mask1 = tok1["attention_mask"]
        mask2 = tok2["attention_mask"]

        for layer_idx, hidden in enumerate(hidden_states):
            hidden1 = hidden[:batch_size]
            hidden2 = hidden[batch_size:]

            dc1 = masked_mean(hidden1, mask1)
            dc2 = masked_mean(hidden2, mask2)

            cosine = F.cosine_similarity(dc1, dc2, dim=-1)
            all_layer_cosines[layer_idx].extend(
                cosine.detach().cpu().numpy().astype(np.float64).tolist()
            )

        all_targets.extend(
            target.detach().cpu().numpy().astype(np.float64).tolist()
        )

    if all_layer_cosines is None:
        raise RuntimeError("Empty loader.")

    return all_layer_cosines, all_targets


def probe_stsb_checkpoint(
    model_path: Path,
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
    checkpoint_index: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    model = AutoModel.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    ckpt_meta = load_stsb_backbone_checkpoint(model, checkpoint_path)
    model.to(device)

    layer_cosines, gold = collect_layer_cosines(model, loader, device)

    seed = infer_seed_from_path(checkpoint_path, checkpoint_index)
    rows = []
    for layer, pred in enumerate(layer_cosines):
        metrics = correlation_metrics(pred, gold)
        rows.append({
            "dataset": "stsb",
            "checkpoint": str(checkpoint_path),
            "seed": seed,
            "layer": layer,
            "layer_name": "Embedding" if layer == 0 else f"Layer {layer}",
            "spearman": metrics["spearman"],
            "pearson": metrics["pearson"],
        })

    final_metrics = rows[-1]
    print(
        f"[STSB seed={seed}] final-layer DC/Mean "
        f"Spearman={final_metrics['spearman']:.6f} "
        f"Pearson={final_metrics['pearson']:.6f}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(rows), {
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        **ckpt_meta,
    }


def probe_sprint(
    model_path: Path,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model = AutoModel.from_pretrained(
        str(model_path),
        local_files_only=True,
    ).to(device)

    # Sprint freezes RoBERTa. Its learned cosine-logit head has positive scale,
    # so it preserves ranking and cannot change Average Precision.
    layer_cosines, labels = collect_layer_cosines(model, loader, device)
    labels_np = np.asarray(labels, dtype=np.int64)

    rows = []
    for layer, cosine in enumerate(layer_cosines):
        ap = float(
            average_precision_score(
                labels_np,
                np.asarray(cosine, dtype=np.float64),
            )
        )
        rows.append({
            "dataset": "sprint",
            "checkpoint": "frozen_roberta_base",
            "seed": np.nan,
            "layer": layer,
            "layer_name": "Embedding" if layer == 0 else f"Layer {layer}",
            "average_precision": ap,
        })

    print(
        "[Sprint] final-layer DC/Mean "
        f"AP={rows[-1]['average_precision']:.6f}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def summarize_stsb(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["layer", "layer_name"], as_index=False)
        .agg(
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            n_checkpoints=("checkpoint", "nunique"),
        )
        .sort_values("layer")
        .reset_index(drop=True)
    )
    summary["spearman_std"] = summary["spearman_std"].fillna(0.0)
    summary["pearson_std"] = summary["pearson_std"].fillna(0.0)
    return summary


def build_combined_table(
    stsb_summary: pd.DataFrame,
    sprint_df: pd.DataFrame,
) -> pd.DataFrame:
    stsb = stsb_summary[
        ["layer", "layer_name", "spearman_mean", "spearman_std"]
    ].copy()
    sprint = sprint_df[["layer", "average_precision"]].copy()

    combined = stsb.merge(sprint, on="layer", how="outer")
    combined = combined.sort_values("layer").reset_index(drop=True)
    combined = combined.rename(
        columns={"average_precision": "sprint_average_precision"}
    )
    return combined


def plot_curves(
    stsb_summary: pd.DataFrame,
    sprint_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    layers = stsb_summary["layer"].to_numpy()
    stsb_mean = stsb_summary["spearman_mean"].to_numpy()
    stsb_std = stsb_summary["spearman_std"].to_numpy()

    sprint_layers = sprint_df["layer"].to_numpy()
    sprint_ap = sprint_df["average_precision"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(layers, stsb_mean, marker="o", label="DC-only / Mean")
    if np.any(stsb_std > 0):
        ax.fill_between(
            layers,
            stsb_mean - stsb_std,
            stsb_mean + stsb_std,
            alpha=0.18,
            label="±1 std across STSB checkpoints",
        )
    ax.set_title("STS-B: layer-wise DC-only semantic similarity")
    ax.set_xlabel("RoBERTa representation")
    ax.set_ylabel("Spearman")
    ax.set_xticks(layers)
    ax.set_xticklabels(
        ["Emb"] + [str(i) for i in range(1, len(layers))]
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(
        sprint_layers,
        sprint_ap,
        marker="o",
        label="DC-only / Mean",
    )
    ax.set_title("Sprint: layer-wise DC-only matching signal")
    ax.set_xlabel("RoBERTa representation")
    ax.set_ylabel("Average Precision")
    ax.set_xticks(sprint_layers)
    ax.set_xticklabels(
        ["Emb"] + [str(i) for i in range(1, len(sprint_layers))]
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.suptitle(
        "Task information readable from the DC / global-mean component",
        fontsize=13,
    )
    fig.tight_layout()

    fig.savefig(
        output_dir / "dc_layer_curve.png",
        dpi=220,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "dc_layer_curve.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe layer-wise DC/mean task information on STS-B and Sprint."
        )
    )
    parser.add_argument(
        "--stsb_checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more STS-B Mean-pooling best_model.pt checkpoints. "
            "Multiple seeds produce mean±std in the STS-B panel."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("/home/data/home/wwr_lumos/models/roberta-base"),
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
            "outputs/text/dc_layer_curve"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)
    print("split:", args.split)

    for path in args.stsb_checkpoints:
        if not path.is_file():
            raise FileNotFoundError(f"STS-B checkpoint not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    print(
        "DC convention: masked mean over all valid tokens, including "
        "RoBERTa special tokens (matches existing MeanPooling)."
    )

    stsb_frames = []
    stsb_checkpoint_meta = []
    for checkpoint_index, checkpoint_path in enumerate(args.stsb_checkpoints):
        frame, meta = probe_stsb_checkpoint(
            model_path=args.model_path,
            checkpoint_path=checkpoint_path,
            loader=stsb_loader,
            device=device,
            checkpoint_index=checkpoint_index,
        )
        stsb_frames.append(frame)
        stsb_checkpoint_meta.append(meta)

    stsb_raw = pd.concat(stsb_frames, ignore_index=True)
    stsb_summary = summarize_stsb(stsb_raw)

    sprint_df = probe_sprint(
        model_path=args.model_path,
        loader=sprint_loader,
        device=device,
    )

    stsb_layers = stsb_summary["layer"].tolist()
    sprint_layers = sprint_df["layer"].tolist()
    if stsb_layers != sprint_layers:
        raise RuntimeError(
            f"Layer mismatch: STSB={stsb_layers}, Sprint={sprint_layers}"
        )

    stsb_raw.to_csv(
        args.output_dir / "stsb_dc_layer_raw.csv",
        index=False,
    )
    stsb_summary.to_csv(
        args.output_dir / "stsb_dc_layer_summary.csv",
        index=False,
    )
    sprint_df.to_csv(
        args.output_dir / "sprint_dc_layer.csv",
        index=False,
    )

    combined = build_combined_table(stsb_summary, sprint_df)
    combined.to_csv(
        args.output_dir / "dc_layer_curve_combined.csv",
        index=False,
    )

    plot_curves(
        stsb_summary=stsb_summary,
        sprint_df=sprint_df,
        output_dir=args.output_dir,
    )

    metadata = {
        "probe": "layer-wise DC-only / masked-mean task-information curve",
        "split": args.split,
        "model_path": str(args.model_path),
        "stsb_data_path": str(args.stsb_data_path),
        "sprint_data_path": str(args.sprint_data_path),
        "stsb_num_pairs": stsb_n,
        "sprint_num_pairs": sprint_n,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": str(device),
        "dc_definition": (
            "masked mean over all valid tokens including special tokens; "
            "proportional to DCT-II/FFT DC"
        ),
        "stsb_metric": "Spearman (Pearson also saved)",
        "sprint_metric": (
            "Average Precision from cosine similarity; identical ranking to "
            "the trained positive-scale affine cosine-logit head"
        ),
        "stsb_checkpoints": stsb_checkpoint_meta,
        "interpretation_warning": (
            "STS-B uses task-finetuned Mean-pooling backbones while Sprint "
            "uses the protocol-specified frozen pretrained RoBERTa backbone. "
            "Compare layerwise shapes/patterns, not raw metric magnitudes."
        ),
    }
    with open(
        args.output_dir / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    for name in [
        "stsb_dc_layer_raw.csv",
        "stsb_dc_layer_summary.csv",
        "sprint_dc_layer.csv",
        "dc_layer_curve_combined.csv",
        "dc_layer_curve.png",
        "dc_layer_curve.pdf",
        "run_metadata.json",
    ]:
        print(" ", args.output_dir / name)

    print("\nFinal-layer sanity targets:")
    print(
        "  STS-B final-layer DC/Mean should reproduce the corresponding "
        "Mean checkpoint's test Spearman/Pearson."
    )
    print(
        "  Sprint final-layer AP should reproduce the Mean model's "
        "cosine_average_precision (positive affine head preserves AP rank)."
    )


if __name__ == "__main__":
    main()
