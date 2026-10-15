#!/usr/bin/env python3
"""Exploratory, hypothesis-neutral Mean-equivalent FLaG trajectory study.

Questions:
  1) Does B2 actually start from the SAME prediction function as raw Mean?
  2) At each checkpoint, does B2's extra pooling help or hurt compared with
     bypassing that pooling using the SAME encoder and SAME score head?
  3) How does this change through training on STSB versus Sprint?

No Gate/Projection freeze, no frequency ablation, no test-set selection.
Original training entrypoints are not modified.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_from_disk
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from AMPCliff.factory.pooling.llm_pooling_dropin import masked_mean_pooling
from train_author_text_protocol import (
    PairCollator, PairDataset, SentencePairModel,
    build_e2e_adam, seed_everything, split_90_10,
)


def get_splits(task, stsb_path, sprint_path, seed, train_limit, val_limit):
    raw = load_from_disk(stsb_path if task == "stsb" else sprint_path)
    if task == "stsb":
        train, val = raw["train"], raw["validation"]
    else:
        # Identical to train_author_text_protocol.py: adaptation validation is
        # recombined, then split by the experiment seed.
        full = concatenate_datasets([raw["train"], raw["validation"]])
        train, val = split_90_10(full, seed)
    if train_limit:
        train = train.select(range(min(train_limit, len(train))))
    if val_limit:
        val = val.select(range(min(val_limit, len(val))))
    return train, val


def get_loaders(task, tokenizer, train_split, val_split, seed):
    collate = PairCollator(tokenizer, max_length=128, label_dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    train = DataLoader(
        PairDataset(train_split, task), batch_size=32,
        shuffle=True, generator=gen, collate_fn=collate, num_workers=0,
    )
    val = DataLoader(
        PairDataset(val_split, task), batch_size=64,
        shuffle=False, collate_fn=collate, num_workers=0,
    )
    return train, val


def metrics(task, labels, logits):
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(logits, dtype=np.float64)
    if task == "stsb":
        target = y / 5.0
        sp = spearmanr(y, s).statistic
        pe = pearsonr(y, s).statistic if len(y) > 1 else float("nan")
        return {
            "spearman": float(sp), "pearson": float(pe),
            "mse": float(np.mean((s - target) ** 2)),
        }
    prob = expit(s)
    return {
        "ap": float(average_precision_score(y.astype(int), prob)),
        "bce": float(np.mean(np.logaddexp(0, s) - y * s)),
        "positive_fraction": float(y.mean()),
    }


@torch.no_grad()
def evaluate(model, loader, task, device):
    model.eval()
    labels, cos_full, cos_mean, scores_full, scores_mean = [], [], [], [], []
    sample_alignment = []
    for toks_a, toks_b, y in loader:
        pair_vec = []
        for toks in (toks_a, toks_b):
            toks = {key: value.to(device) for key, value in toks.items()}
            mask = toks["attention_mask"]
            hidden = model.encoder(**toks).last_hidden_state
            full = F.normalize(
                model.pool(hidden, attention_mask=mask), p=2, dim=-1,
            )
            simple = F.normalize(
                masked_mean_pooling(hidden, mask, eps=1e-6), p=2, dim=-1,
            )
            sample_alignment.append(
                (full * simple).sum(dim=-1).detach().cpu().numpy()
            )
            pair_vec.append((full, simple))

        (fa, ma), (fb, mb) = pair_vec
        cf = (fa * fb).sum(dim=-1)
        cm = (ma * mb).sum(dim=-1)
        factor = model.log_scale.detach().exp()
        offset = model.bias.detach()
        scores_full.extend((factor * cf + offset).cpu().tolist())
        scores_mean.extend((factor * cm + offset).cpu().tolist())
        cos_full.extend(cf.cpu().tolist())
        cos_mean.extend(cm.cpu().tolist())
        labels.extend(y.tolist())

    arrays = {
        "labels": np.asarray(labels, dtype=np.float32),
        "cos_full": np.asarray(cos_full, dtype=np.float32),
        "cos_same_encoder_mean": np.asarray(cos_mean, dtype=np.float32),
        "score_full": np.asarray(scores_full, dtype=np.float32),
        "score_same_encoder_mean": np.asarray(scores_mean, dtype=np.float32),
    }
    angle_cos = np.concatenate(sample_alignment)
    result = {
        "full": metrics(task, arrays["labels"], arrays["score_full"]),
        "same_encoder_mean": metrics(
            task, arrays["labels"], arrays["score_same_encoder_mean"],
        ),
        "mean_abs_pair_cos_delta": float(
            np.mean(np.abs(arrays["cos_full"] - arrays["cos_same_encoder_mean"]))
        ),
        "mean_sentence_cos_to_same_encoder_mean": float(angle_cos.mean()),
        "max_abs_pair_cos_delta": float(
            np.max(np.abs(arrays["cos_full"] - arrays["cos_same_encoder_mean"]))
        ),
        "head_scale": float(model.log_scale.detach().exp().cpu()),
        "head_bias": float(model.bias.detach().cpu()),
        "n_validation_pairs": int(len(labels)),
    }
    return result, arrays


def epoch_train(model, loader, task, device, optimizer):
    model.train()
    total_loss, count = 0.0, 0
    for toks_a, toks_b, y in loader:
        a = {key: val.to(device) for key, val in toks_a.items()}
        b = {key: val.to(device) for key, val in toks_b.items()}
        labels = y.to(device)
        target = labels / 5.0 if task == "stsb" else labels
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(a, b)
        loss = (F.mse_loss(logits, target) if task == "stsb"
                else F.binary_cross_entropy_with_logits(logits, target))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(y)
        count += len(y)
    return total_loss / max(1, count)


def run_one(args, task, seed, mode, tokenizer, device):
    output = Path(args.output_dir) / task / mode / f"seed_{seed}"
    if output.exists() and (output / "complete.json").exists() and not args.force:
        print(f"[skip complete] {output}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    train_split, val_split = get_splits(
        task, args.stsb_data_path, args.sprint_data_path,
        seed, args.train_limit, args.val_limit,
    )
    train_loader, val_loader = get_loaders(
        task, tokenizer, train_split, val_split, seed,
    )

    seed_everything(seed)
    model = SentencePairModel(
        args.model_path,
        "mean" if mode == "mean" else "FLaG_B2",
        freeze_backbone=(task == "sprint"),
        mean_anchor_beta_init=0.1,
        flag_time_pool="mean",
        flag_post_pool_norm=False,
    ).to(device)

    optimizer = (
        torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-3, weight_decay=0.0,
        )
        if task == "sprint"
        else build_e2e_adam(model, model.encoder, 1e-5, 1e-3)
    )

    epoch_count = args.max_epochs or (3 if task == "stsb" else 10)
    best_value, best_epoch = -float("inf"), None
    criterion = "spearman" if task == "stsb" else "ap"
    history = []
    # Reset after network construction: the same-seed RoBERTa initial weights
    # and identical DataLoader generator are used in both compared runs.
    seed_everything(seed + 100000)
    for epoch in range(epoch_count + 1):
        train_loss = None
        if epoch:
            train_loss = epoch_train(model, train_loader, task, device, optimizer)
        val, arrays = evaluate(model, val_loader, task, device)
        record = {
            "task": task, "seed": seed, "pooling": mode, "epoch": epoch,
            "online_epoch_train_loss": train_loss,
            **val,
        }
        history.append(record)
        np.savez_compressed(output / f"val_pairs_epoch_{epoch}.npz", **arrays)
        with (output / "trajectory.jsonl").open("w", encoding="utf-8") as fh:
            for row in history:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        metric = val["full"][criterion]
        if np.isfinite(metric) and metric > best_value:
            best_epoch, best_value = epoch, metric
            if args.save_checkpoints:
                torch.save(model.state_dict(), output / "best_model.pt")
        print(
            f"[{task}][{mode}][seed={seed}][epoch={epoch}] "
            f"val_{criterion}={metric:.6f} "
            f"same_encoder_mean_{criterion}={val['same_encoder_mean'][criterion]:.6f} "
            f"pair_cos_delta={val['mean_abs_pair_cos_delta']:.8f} "
            f"train_loss={train_loss}", flush=True,
        )
    if args.save_checkpoints:
        torch.save(model.state_dict(), output / "last_model.pt")
    with (output / "complete.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "task": task, "mode": mode, "seed": seed,
            "best_epoch": best_epoch, "best_val_metric": best_value,
            "criterion": criterion,
            "n_train_pairs": len(train_split),
            "n_val_pairs": len(val_split),
            "epochs": epoch_count,
            "backbone_frozen": task == "sprint",
            "checkpoint_selection": "validation " + criterion,
            "evaluation_split": "validation only; test untouched",
            "training_details": "batch32, Adam, wd=0, pool_lr=1e-3, "
                "STSB backbone_lr=1e-5, no post pool LN, mean time pool",
        }, fh, indent=2)
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compare_runs(output_dir, task, seed):
    root = Path(output_dir) / task
    paths = [root / mode / f"seed_{seed}" for mode in ("mean", "flag_b2")]
    if not all((p / "complete.json").exists() for p in paths):
        return
    def rows(path):
        return [
            json.loads(line) for line in (path / "trajectory.jsonl").read_text().splitlines()
        ]
    mean, flag = (rows(p) for p in paths)
    metric = "spearman" if task == "stsb" else "ap"
    paired = []
    for m, f in zip(mean, flag):
        assert m["epoch"] == f["epoch"]
        row = {
            "epoch": f["epoch"],
            "independently_trained_mean": m["full"][metric],
            "flag_b2": f["full"][metric],
            "flag_checkpoint_mean": f["same_encoder_mean"][metric],
            "flag_minus_same_encoder_mean":
                f["full"][metric] - f["same_encoder_mean"][metric],
            "flag_minus_independent_mean":
                f["full"][metric] - m["full"][metric],
            "flag_pair_cos_change": f["mean_abs_pair_cos_delta"],
        }
        if f["epoch"] == 0:
            ma = np.load(paths[0] / "val_pairs_epoch_0.npz")
            fa = np.load(paths[1] / "val_pairs_epoch_0.npz")
            row["initial_max_abs_cos_mean_vs_b2"] = float(
                np.max(np.abs(ma["cos_full"] - fa["cos_full"]))
            )
            if row["initial_max_abs_cos_mean_vs_b2"] > 2e-4:
                raise RuntimeError(
                    "B2 and Mean NOT equal at epoch 0: "
                    + str(row["initial_max_abs_cos_mean_vs_b2"])
                )
        paired.append(row)
    out = root / f"seed_{seed}_comparison.json"
    out.write_text(json.dumps(paired, ensure_ascii=False, indent=2))
    print(f"[comparison saved] {out}", flush=True)
    for row in paired:
        print(
            f"  ep={row['epoch']:>2}  independent_mean={row['independently_trained_mean']:.6f} "
            f"B2={row['flag_b2']:.6f} "
            f"B2_checkpoint_mean={row['flag_checkpoint_mean']:.6f} "
            f"delta_cos={row['flag_pair_cos_change']:.8f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=["stsb", "sprint"],
                        default=["stsb", "sprint"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--model_path",
                        default="/home/data/home/wwr_lumos/models/roberta-base")
    parser.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    parser.add_argument("--sprint_data_path",
                        default="data/text/sprintduplicatequestions_adaptation")
    parser.add_argument("--output_dir",
                        default="outputs/text/mean_equiv_exploration")
    parser.add_argument("--max_epochs", type=int, default=0,
                        help="0 = official 3 STSB / 10 Sprint; for smoke tests only")
    parser.add_argument("--train_limit", type=int, default=0,
                        help="0 = full; positive values for smoke tests only")
    parser.add_argument("--val_limit", type=int, default=0,
                        help="0 = full; positive values for smoke tests only")
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="Save best and last full state_dict; uses substantial disk")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True,
    )
    for task in args.tasks:
        for seed in args.seeds:
            for mode in ("mean", "flag_b2"):
                run_one(args, task, seed, mode, tokenizer, device)
            compare_runs(args.output_dir, task, seed)


if __name__ == "__main__":
    main()
