#!/usr/bin/env python3
"""Isolated STSB Pairwise-Hinge training audit; never changes prior experiment outputs.

Stage 1: --mode subset --model mean (can Hinge memorize 128 examples?)
Stage 2: --mode full   --model mean (does 3-epoch training stop too early?)
Only if Mean passes: repeat with --model flag_zero.

No test-set evaluation is performed. This is a diagnostic, not a new benchmark.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

# The training modules use absolute imports like `AMPCliff.factory...`.
# When launched as `python text_repro/diagnose_stsb_pairwise_hinge.py`, Python
# adds `text_repro` to sys.path, but not the parent of the AMPCliff package.
# Add the checkout's parent BEFORE importing train_author_text_protocol.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from train_author_text_protocol import (
    PairCollator, PairDataset, build_e2e_adam, seed_everything
)
from run_stsb_ranknet_comparison import (
    check_flag_zero, evaluate, make_model, pairwise_hinge_loss
)


def test_hinge_direction():
    """Check expected ordering yields smaller loss than reversed ordering."""
    labels = torch.tensor([0.0, 1.0, 2.0])
    good = torch.tensor([0.0, 2.0, 4.0], requires_grad=True)
    bad = torch.tensor([4.0, 2.0, 0.0], requires_grad=True)
    good_loss, n_good = pairwise_hinge_loss(good, labels, margin=1.0)
    bad_loss, n_bad = pairwise_hinge_loss(bad, labels, margin=1.0)
    assert n_good == 3 and n_bad == 3
    assert float(good_loss) == 0.0 and float(bad_loss) > 0.0
    bad_loss.backward()
    assert bad.grad is not None and torch.isfinite(bad.grad).all().item()
    print("[self-test] Pairwise Hinge ordering and gradients PASS", flush=True)


def pair_metrics(raw_scores, raw_labels):
    """Pairwise accuracy and Hinge margin violations on a diagnostic sample."""
    s = np.asarray(raw_scores, dtype=np.float64)
    y = np.asarray(raw_labels, dtype=np.float64)
    i, j = np.triu_indices(len(y), 1)
    delta = y[i] - y[j]
    valid = np.abs(delta) > 1e-8
    signed = np.sign(delta[valid]) * (s[i[valid]] - s[j[valid]])
    if signed.size == 0:
        return dict(pair_accuracy=None, margin_violation_fraction=None,
                    avg_signed_gap=None, valid_pairs=0)
    correct = (signed > 0).astype(np.float64) + 0.5 * (signed == 0)
    return dict(
        pair_accuracy=float(correct.mean()),
        margin_violation_fraction=float(np.mean(signed < 1.0)),
        avg_signed_gap=float(signed.mean()),
        valid_pairs=int(signed.size),
    )


def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().float().square().sum().item())
    return math.sqrt(total)


def build_loader(data, tokenizer, batch_size, shuffle, seed):
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=PairCollator(tokenizer, max_length=128, label_dtype=torch.float32),
        num_workers=0,
    )


def eval_sample(model, loader, device):
    stats, arrays = evaluate(model, loader, device)
    pair = pair_metrics(arrays["scores"], arrays["labels"])
    return stats, pair


def write_history(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    test_hinge_direction()
    seed_everything(args.seed)
    epochs = args.epochs if args.epochs is not None else (25 if args.mode == "subset" else 12)
    root = Path(args.out_root)
    tag = "n" + str(args.subset_size) if args.mode == "subset" else "all"
    out = root / args.model / args.mode / ("seed_" + str(args.seed) + "_" + tag + "_e" + str(epochs))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Refusing to overwrite diagnostics: " + str(out))
    out.mkdir(parents=True, exist_ok=True)

    raw = load_from_disk(args.stsb_data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    all_train = PairDataset(raw["train"], "stsb")
    val_ds = PairDataset(raw["validation"], "stsb")
    sample_rng = np.random.default_rng(args.seed)
    if args.mode == "subset":
        ids = sample_rng.choice(len(all_train), size=min(args.subset_size, len(all_train)), replace=False)
        train_ds = Subset(all_train, sorted(int(x) for x in ids))
        eval_train_ds = train_ds
    else:
        train_ds = all_train
        # Pairwise metrics on at most 512 fixed training examples to bound O(n^2) cost.
        ids = sample_rng.choice(len(all_train), size=min(512, len(all_train)), replace=False)
        eval_train_ds = Subset(all_train, sorted(int(x) for x in ids))
    train_loader = build_loader(train_ds, tokenizer, args.batch_size, True, args.seed + 1000)
    train_eval = build_loader(eval_train_ds, tokenizer, 64, False, args.seed + 2000)
    val_loader = build_loader(val_ds, tokenizer, 64, False, args.seed + 3000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("GPU not detected; pass --allow_cpu only if CPU execution is intentional.")
    model = make_model(args.model, args, device)
    if args.model == "flag_zero":
        initial_equivalence = check_flag_zero(model, next(iter(train_eval)), device)
    else:
        initial_equivalence = {"applicable": False}

    optimizer = build_e2e_adam(model, model.encoder, args.backbone_lr, args.new_lr)
    backbone = list(model.encoder.parameters())
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    n_backbone = sum(p.numel() for p in backbone if p.requires_grad)
    n_head = sum(p.numel() for p in head if p.requires_grad)
    print(f"mode={args.mode} model={args.model} seed={args.seed} epochs={epochs} "
          f"train_examples={len(train_ds)} diagnostics_train_examples={len(eval_train_ds)} "
          f"device={device} trainable_backbone={n_backbone} trainable_new={n_head}", flush=True)
    print(f"out={out} initial_equivalence={initial_equivalence}", flush=True)

    history = []
    best_val_sp = -float("inf")
    best_val_epoch = None
    for epoch in range(1, epochs + 1):
        model.train()
        sum_loss, sum_pairs = 0.0, 0
        backbone_grad, head_grad, updates = 0.0, 0.0, 0
        for t1, t2, labels in train_loader:
            t1 = {k: v.to(device) for k, v in t1.items()}
            t2 = {k: v.to(device) for k, v in t2.items()}
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(t1, t2)
            loss, pairs = pairwise_hinge_loss(pred, labels, margin=1.0)
            if pairs == 0:
                continue
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Nonfinite Hinge loss")
            loss.backward()
            backbone_grad += grad_norm(backbone)
            head_grad += grad_norm(head)
            if not math.isfinite(backbone_grad + head_grad):
                raise RuntimeError("Nonfinite gradient")
            optimizer.step()
            updates += 1
            sum_loss += float(loss.detach()) * pairs
            sum_pairs += pairs

        if not updates:
            raise RuntimeError("No parameter updates; every training minibatch had only tied labels.")
        row = {
            "epoch": epoch,
            "train_hinge_loss": sum_loss / max(sum_pairs, 1),
            "train_pairs": sum_pairs,
            "grad_backbone_mean": backbone_grad / updates,
            "grad_new_mean": head_grad / updates,
            "scale": float(torch.exp(model.log_scale).detach().cpu()),
            "bias": float(model.bias.detach().cpu()),
            "train_spearman_sample": "",
            "train_pair_accuracy": "",
            "train_margin_violation_fraction": "",
            "train_avg_signed_gap": "",
            "valid_train_pairs": "",
            "val_spearman": "",
        }
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == epochs:
            train_stats, pair = eval_sample(model, train_eval, device)
            val_stats, _ = evaluate(model, val_loader, device)
            row.update(
                train_spearman_sample=train_stats["spearman"],
                train_pair_accuracy=pair["pair_accuracy"],
                train_margin_violation_fraction=pair["margin_violation_fraction"],
                train_avg_signed_gap=pair["avg_signed_gap"],
                valid_train_pairs=pair["valid_pairs"],
                val_spearman=val_stats["spearman"],
            )
            if math.isfinite(val_stats["spearman"]) and val_stats["spearman"] > best_val_sp:
                best_val_sp, best_val_epoch = val_stats["spearman"], epoch
        history.append(row)
        # Save each epoch so an interrupted GPU session still leaves a useful trajectory.
        write_history(out / "history.csv", history)
        print(
            f"epoch={epoch:02d} hinge={row['train_hinge_loss']:.5f} "
            f"scale={row['scale']:.3f} grad_backbone={row['grad_backbone_mean']:.4f} "
            f"train_sp={row['train_spearman_sample']} pair_acc={row['train_pair_accuracy']} "
            f"margin_violation={row['train_margin_violation_fraction']} val_sp={row['val_spearman']}",
            flush=True,
        )

    report = {
        "protocol": "Single-loss Pairwise-Hinge diagnosis only; not benchmark",
        "model": args.model, "mode": args.mode, "seed": args.seed, "epochs": epochs,
        "batch_size": args.batch_size, "subset_size": args.subset_size if args.mode == "subset" else None,
        "train_size": len(train_ds),
        "train_diagnostic_size": len(eval_train_ds),
        "best_observed_val_spearman": best_val_sp,
        "best_observed_val_epoch": best_val_epoch,
        "last_eval": history[-1],
        "initial_equivalence": initial_equivalence,
        "optimizer": "Adam",
        "backbone_lr": args.backbone_lr, "new_lr": args.new_lr,
        "loss": "mean(max(0, 1-sign(y_i-y_j)*(score_i-score_j))) over non-tied pairs",
        "test_set_evaluated": False,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("DIAGNOSIS_RESULT " + json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("subset", "full"), default="subset")
    parser.add_argument("--model", choices=("mean", "flag_zero"), default="mean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--subset_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--new_lr", type=float, default=1e-3)
    parser.add_argument("--model_path", default="/home/data/home/wwr_lumos/models/roberta-base")
    parser.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    parser.add_argument("--out_root", default="outputs/text/stsb_ltr_hinge_diagnostic")
    parser.add_argument("--allow_cpu", action="store_true")
    args = parser.parse_args()
    if args.eval_every <= 0 or args.subset_size < 2 or args.batch_size < 2:
        parser.error("eval_every > 0, subset_size >= 2, and batch_size >= 2 required")
    run(args)


if __name__ == "__main__":
    main()
