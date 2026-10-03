#!/usr/bin/env python3
"""Sweep the learned Mean/FLaG mixing scalar alpha without retraining.

For a trained FLaG_MeanResidual checkpoint, evaluate the SAME learned
Mean branch + FLaG branch while forcing alpha over a fixed grid.

This separates:
  (a) whether the trained branches are useful,
from
  (b) whether gradient descent found a good scalar alpha.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = _REPO_ROOT.parent
for _p in (_REPO_PARENT, _REPO_ROOT / "text_repro"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from train_sts import (
    STSCollator,
    STSDataset,
    SentenceEncoder,
    evaluate as evaluate_stsb,
)
from train_sprint import (
    SprintCollator,
    SprintDataset,
    SprintPairClassifier,
    evaluate as evaluate_sprint,
)


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as f:
        return json.load(f)


def make_loader(dataset, batch_size, collator):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )


def load_stsb(seed: int, device):
    run_dir = Path(
        "outputs/text/stsbenchmark/experiments/"
        f"stsb_unfrozen_flag_meanresidual/seed_{seed}"
    )
    cfg = read_json(run_dir / "config.json")
    ckpt = torch.load(
        run_dir / "best_model.pt",
        map_location=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_path"],
        local_files_only=True,
    )
    raw = load_from_disk(cfg["data_path"])
    collator = STSCollator(
        tokenizer,
        max_length=int(cfg["max_length"]),
    )

    model = SentenceEncoder(
        cfg["model_path"],
        "FLaG_MeanResidual",
        stft_win_length=int(cfg["stft_win_length"]),
        stft_hop_length=int(cfg["stft_hop_length"]),
        stft_window_type=cfg["stft_window_type"],
        stft_center=bool(cfg["stft_center"]),
        fixed_fft_length=cfg["fixed_fft_length"],
        pool_dropout=float(cfg["pool_dropout"]),
        post_pool_norm=bool(cfg["post_pool_norm"]),
        freeze_backbone=bool(cfg["freeze_backbone"]),
        remove_dc=bool(cfg["remove_dc"]),
        dc_only=bool(cfg["dc_only"]),
        mean_mix_init=float(cfg["mean_mix_init"]),
    ).to(device)
    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )
    model.eval()

    val_loader = make_loader(
        STSDataset(raw["validation"]),
        int(cfg["batch_size"]),
        collator,
    )
    test_loader = make_loader(
        STSDataset(raw["test"]),
        int(cfg["batch_size"]),
        collator,
    )

    return model, val_loader, test_loader, run_dir


def load_sprint(seed: int, device):
    run_dir = Path(
        "outputs/text/sprintduplicatequestions/experiments/"
        f"sprint_frozen_flag_meanresidual/seed_{seed}"
    )
    cfg = read_json(run_dir / "config.json")
    ckpt = torch.load(
        run_dir / "best_model.pt",
        map_location=device,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_path"],
        local_files_only=True,
    )
    raw = load_from_disk(cfg["data_path"])
    collator = SprintCollator(
        tokenizer,
        max_length=int(cfg["max_length"]),
    )

    model = SprintPairClassifier(
        model_path=cfg["model_path"],
        pooling="FLaG_MeanResidual",
        stft_win_length=int(cfg["stft_win_length"]),
        stft_hop_length=int(cfg["stft_hop_length"]),
        stft_window_type=cfg["stft_window_type"],
        stft_center=bool(cfg["stft_center"]),
        finetune_backbone=bool(cfg["finetune_backbone"]),
        remove_dc=bool(cfg["remove_dc"]),
        dc_only=bool(cfg["dc_only"]),
        mean_mix_init=float(cfg["mean_mix_init"]),
    ).to(device)
    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )
    model.eval()

    val_loader = make_loader(
        SprintDataset(raw["validation"]),
        int(cfg["eval_batch_size"]),
        collator,
    )
    test_loader = make_loader(
        SprintDataset(raw["test"]),
        int(cfg["eval_batch_size"]),
        collator,
    )

    return model, val_loader, test_loader, run_dir


@torch.no_grad()
def sweep_stsb(model, val_loader, test_loader, device, alphas):
    rows = []

    learned_alpha = float(
        torch.clamp(
            model.pool.mean_mix_alpha.detach(),
            0.0,
            1.0,
        ).cpu()
    )

    for alpha in alphas:
        model.pool.mean_mix_alpha.fill_(float(alpha))

        val_metrics, _, _ = evaluate_stsb(
            model,
            val_loader,
            device,
        )
        test_metrics, _, _ = evaluate_stsb(
            model,
            test_loader,
            device,
        )

        rows.append({
            "task": "STSB",
            "alpha": float(alpha),
            "val_spearman": float(val_metrics["spearman"]),
            "val_pearson": float(val_metrics["pearson"]),
            "test_spearman": float(test_metrics["spearman"]),
            "test_pearson": float(test_metrics["pearson"]),
        })

        print(
            f"STSB alpha={alpha:.2f} | "
            f"val Spearman={val_metrics['spearman']:.6f} | "
            f"test Spearman={test_metrics['spearman']:.6f}"
        )

    model.pool.mean_mix_alpha.fill_(learned_alpha)
    return rows, learned_alpha


@torch.no_grad()
def sweep_sprint(model, val_loader, test_loader, device, alphas):
    rows = []

    learned_alpha = float(
        torch.clamp(
            model.pool.mean_mix_alpha.detach(),
            0.0,
            1.0,
        ).cpu()
    )

    for alpha in alphas:
        model.pool.mean_mix_alpha.fill_(float(alpha))

        val_metrics, _, _, _ = evaluate_sprint(
            model,
            val_loader,
            device,
        )
        test_metrics, _, _, _ = evaluate_sprint(
            model,
            test_loader,
            device,
        )

        rows.append({
            "task": "Sprint",
            "alpha": float(alpha),
            "val_ap": float(val_metrics["average_precision"]),
            "val_cosine_ap": float(
                val_metrics["cosine_average_precision"]
            ),
            "test_ap": float(test_metrics["average_precision"]),
            "test_cosine_ap": float(
                test_metrics["cosine_average_precision"]
            ),
        })

        print(
            f"Sprint alpha={alpha:.2f} | "
            f"val AP={val_metrics['average_precision']:.6f} | "
            f"test AP={test_metrics['average_precision']:.6f}"
        )

    model.pool.mean_mix_alpha.fill_(learned_alpha)
    return rows, learned_alpha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[i / 10 for i in range(11)],
    )
    p.add_argument("--skip_stsb", action="store_true")
    p.add_argument("--skip_sprint", action="store_true")
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "outputs/text/mean_residual_alpha_sweep"
        ),
    )
    args = p.parse_args()

    alphas = sorted(set(float(a) for a in args.alphas))
    for a in alphas:
        if not (0.0 <= a <= 1.0):
            raise ValueError(
                f"alpha must be in [0,1], got {a}"
            )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)
    print("seed:", args.seed)
    print("alphas:", alphas)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.skip_stsb:
        print("\n" + "=" * 82)
        print("STSB alpha sweep | UNFROZEN original protocol")
        print("=" * 82)

        model, val_loader, test_loader, _ = load_stsb(
            args.seed,
            device,
        )
        rows, learned_alpha = sweep_stsb(
            model,
            val_loader,
            test_loader,
            device,
            alphas,
        )
        df = pd.DataFrame(rows)
        out = args.output_dir / (
            f"stsb_alpha_sweep_seed{args.seed}.csv"
        )
        df.to_csv(out, index=False)

        best = df.loc[
            df["val_spearman"].idxmax()
        ]
        print(
            f"\nSTSB learned alpha={learned_alpha:.6f}"
        )
        print(
            "STSB best grid alpha by validation Spearman="
            f"{best['alpha']:.2f} | "
            f"val={best['val_spearman']:.6f} | "
            f"test={best['test_spearman']:.6f}"
        )
        print("Saved:", out)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not args.skip_sprint:
        print("\n" + "=" * 82)
        print("Sprint alpha sweep | FROZEN original protocol")
        print("=" * 82)

        model, val_loader, test_loader, _ = load_sprint(
            args.seed,
            device,
        )
        rows, learned_alpha = sweep_sprint(
            model,
            val_loader,
            test_loader,
            device,
            alphas,
        )
        df = pd.DataFrame(rows)
        out = args.output_dir / (
            f"sprint_alpha_sweep_seed{args.seed}.csv"
        )
        df.to_csv(out, index=False)

        best = df.loc[
            df["val_ap"].idxmax()
        ]
        print(
            f"\nSprint learned alpha={learned_alpha:.6f}"
        )
        print(
            "Sprint best grid alpha by validation AP="
            f"{best['alpha']:.2f} | "
            f"val={best['val_ap']:.6f} | "
            f"test={best['test_ap']:.6f}"
        )
        print("Saved:", out)


if __name__ == "__main__":
    main()
