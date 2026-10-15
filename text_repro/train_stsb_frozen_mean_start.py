#!/usr/bin/env python3
"""STSB pilot: start FLaG_B2 at a GOOD, trained Mean solution and freeze RoBERTa.

This experiment deliberately does not claim a mechanism in advance.
Mean reference is fully fine-tuned on STSB and selected by VALIDATION Spearman.
Transfer its exact RoBERTa + cosine-score head to FLaG_B2, check initial
identity, freeze all encoder weights, then train only FLaG pooling and head.

Question: with a fixed, already-useful Mean representation available at
initialization, does FLaG stay good, improve, or degrade during training?

This is a new, controlled B2 experiment, NOT reproduction of original FLaG.
No test-set evaluation and no edits to existing training entrypoints.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from AMPCliff.factory.pooling.llm_pooling_dropin import masked_mean_pooling
from explore_mean_equiv_trajectory import (
    epoch_train, evaluate, get_loaders, get_splits,
)
from train_author_text_protocol import (
    SentencePairModel, build_e2e_adam, seed_everything,
)


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def make_mean_model(args, device):
    return SentencePairModel(
        args.model_path, "mean", freeze_backbone=False,
        mean_anchor_beta_init=0.1, flag_time_pool="mean",
        flag_post_pool_norm=False,
    ).to(device)


def fit_mean_reference(args, seed, tokenizer, device, train_data, val_data,
                       checkpoint, out):
    """Match the earlier seed-0 mean trajectory (STSB, 3 epochs, Adam)."""
    print("[reference] No saved trained Mean weights found; "
          "fine-tuning Mean from RoBERTa-base first.", flush=True)
    train_loader, val_loader = get_loaders(
        "stsb", tokenizer, train_data, val_data, seed,
    )
    seed_everything(seed)
    mean = make_mean_model(args, device)
    opt = build_e2e_adam(mean, mean.encoder, 1e-5, 1e-3)
    seed_everything(seed + 100000)
    best = -float("inf")
    history = []
    for epoch in range(1, args.reference_epochs + 1):
        train_mse = epoch_train(mean, train_loader, "stsb", device, opt)
        evaluation, _ = evaluate(mean, val_loader, "stsb", device)
        val = evaluation["full"]
        record = {
            "epoch": epoch, "online_train_mse": train_mse,
            "val_spearman": val["spearman"],
            "val_pearson": val["pearson"],
            "val_mse": val["mse"],
        }
        history.append(record)
        print(
            f"[reference Mean][epoch {epoch}] train_mse={train_mse:.6f} "
            f"val_spearman={val['spearman']:.6f} val_mse={val['mse']:.6f}",
            flush=True,
        )
        if np.isfinite(val["spearman"]) and val["spearman"] > best:
            best = val["spearman"]
            torch.save(mean.state_dict(), checkpoint)
    save_json(out / "mean_reference_training_history.json", history)
    del mean, opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not checkpoint.is_file():
        raise RuntimeError("Mean reference training produced no valid checkpoint")
    return checkpoint, "trained_mean_reference"


def locate_reference(args, seed, output):
    if args.mean_checkpoint:
        path = Path(args.mean_checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit Mean checkpoint missing: {path}")
        return path, "explicit_mean_checkpoint"

    # Option 1: existing trained Mean checkpoint from the previous exploration,
    # only present when that experiment used SAVE_CHECKPOINTS=1.
    previous = (
        Path("outputs/text/mean_equiv_exploration")
        / "stsb" / "mean" / f"seed_{seed}" / "best_model.pt"
    )
    if previous.is_file():
        return previous, "previous_mean_equiv_exploration"

    # Option 2: this pilot's previously fitted and saved reference.
    local = output / "mean_reference_best.pt"
    if local.is_file():
        return local, "previous_local_mean_reference"
    return local, "missing"


def load_and_evaluate_reference(args, path, source, device, val_loader):
    state = torch.load(path, map_location="cpu", weights_only=True)
    # We only accept the Mean SentencePairModel state dict. There should be
    # encoder and score-head parameters, no trained FLaG pool weights.
    if not isinstance(state, dict) or not any(
        key.startswith("encoder.") for key in state
    ):
        raise ValueError(f"Not a Mean SentencePairModel state_dict: {path}")
    if any(key.startswith("pool.") for key in state):
        raise ValueError("Checkpoint has a learned pooling module; expected Mean")

    ref = make_mean_model(args, device)
    ref.load_state_dict(state, strict=True)
    result, arrays = evaluate(ref, val_loader, "stsb", device)
    print(
        f"[reference] source={source} checkpoint={path}\n"
        f"[reference] Validation Spearman={result['full']['spearman']:.6f}, "
        f"Pearson={result['full']['pearson']:.6f}, "
        f"MSE={result['full']['mse']:.6f}",
        flush=True,
    )
    if result["full"]["spearman"] < args.min_reference_spearman:
        raise RuntimeError(
            "The chosen Mean reference is not sufficiently good to test "
            "retaining a good Mean solution: "
            f"{result['full']['spearman']:.6f} < "
            f"{args.min_reference_spearman:.6f}. "
            "Check checkpoint and protocol, or explicitly change the "
            "--min_reference_spearman threshold for a smoke test."
        )
    del ref
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state, result, arrays


def build_frozen_b2(args, device, ref_state):
    seed_everything(args.seed)
    model = SentencePairModel(
        args.model_path, "FLaG_B2", freeze_backbone=True,
        mean_anchor_beta_init=0.1,
        flag_time_pool="mean", flag_post_pool_norm=False,
    ).to(device)
    encoder_state = {
        key[len("encoder."):]: value
        for key, value in ref_state.items()
        if key.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_state, strict=True)
    with torch.no_grad():
        model.log_scale.copy_(ref_state["log_scale"].to(device))
        model.bias.copy_(ref_state["bias"].to(device))
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.pool.parameters())
    model.encoder.eval()
    return model


@torch.no_grad()
def assert_initial_pool_identity(model, val_loader, device):
    model.eval()
    tokens_a, _, _ = next(iter(val_loader))
    tokens_a = {k: v.to(device) for k, v in tokens_a.items()}
    mask = tokens_a["attention_mask"]
    hidden = model.encoder(**tokens_a).last_hidden_state
    actual = model.pool(hidden, attention_mask=mask)
    expected = masked_mean_pooling(hidden, mask, eps=1e-6)
    max_difference = float((actual - expected).abs().max().cpu())
    print(f"[initial identity] max_abs_pool_vector_difference="
          f"{max_difference:.3e}", flush=True)
    if max_difference > 2e-5:
        raise RuntimeError(
            "B2 does not reproduce Mean after weight transfer. "
            f"max_abs_diff={max_difference:.3e}"
        )
    return max_difference


def pool_and_head_snapshot(model, reference_path, epoch, step):
    # Save only extra learned layers to avoid duplicating a 500MB frozen
    # RoBERTa checkpoint. Reconstruct from the referenced Mean checkpoint.
    return {
        "reference_checkpoint": str(reference_path),
        "epoch": epoch, "step": step,
        "pool": {k: v.detach().cpu() for k, v in model.pool.state_dict().items()},
        "log_scale": model.log_scale.detach().cpu(),
        "bias": model.bias.detach().cpu(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_path",
                    default="/home/data/home/wwr_lumos/models/roberta-base")
    ap.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    ap.add_argument("--output_dir",
                    default="outputs/text/stsb_frozen_trained_mean")
    ap.add_argument("--mean_checkpoint", default="",
                    help="Optional existing trained STSB Mean state_dict")
    ap.add_argument("--min_reference_spearman", type=float, default=0.86)
    ap.add_argument("--reference_epochs", type=int, default=3)
    ap.add_argument("--flag_epochs", type=int, default=3)
    ap.add_argument("--probe_steps", nargs="*", type=int, default=[1, 10, 50],
                    help="Global optimizer steps at which to validate")
    ap.add_argument("--train_limit", type=int, default=0,
                    help="Smoke testing only; full scientific run uses 0")
    ap.add_argument("--val_limit", type=int, default=0,
                    help="Smoke testing only; full scientific run uses 0")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir) / f"seed_{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists() and not args.force:
        print(f"[skip] Complete result exists: {out / 'summary.json'}\n"
              "Use --force to run again.", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Frozen RoBERTa from a well-trained Mean / STSB seed {args.seed} ===")
    print(f"device={device} output={out}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True,
    )
    train_data, val_data = get_splits(
        "stsb", args.stsb_data_path, "", args.seed,
        args.train_limit, args.val_limit,
    )
    checkpoint, origin = locate_reference(args, args.seed, out)
    if origin == "missing":
        checkpoint, origin = fit_mean_reference(
            args, args.seed, tokenizer, device,
            train_data, val_data, checkpoint, out,
        )
    else:
        print(f"[reference] Reusing {checkpoint}", flush=True)

    # Fresh loader with the original seed, independent of any reference
    # fine-tuning that took place above.
    train_loader, val_loader = get_loaders(
        "stsb", tokenizer, train_data, val_data, args.seed,
    )
    ref_state, ref_eval, ref_arrays = load_and_evaluate_reference(
        args, checkpoint, origin, device, val_loader,
    )
    save_json(out / "mean_reference_validation.json", {
        "source": origin, "checkpoint": str(checkpoint),
        "metrics": ref_eval["full"], "seed": args.seed,
        "n_train": len(train_data), "n_validation": len(val_data),
    })
    np.savez_compressed(out / "mean_reference_val_pairs.npz", **ref_arrays)

    model = build_frozen_b2(args, device, ref_state)
    del ref_state
    gc.collect()
    pool_initial_max_error = assert_initial_pool_identity(
        model, val_loader, device,
    )
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=0.0,
    )
    # No RoBERTa updates. Use a deterministic seed for any stochastic
    # behavior in the newly added B2 components; B2 dropout is zero.
    seed_everything(args.seed + 100000)
    history = []
    best_step, best_metric = None, -float("inf")
    last_probe_step = None
    ref_sp = ref_eval["full"]["spearman"]
    init_score_error = None
    global_step = 0

    def snapshot(epoch, phase, last_batch_loss=None):
        nonlocal best_step, best_metric, last_probe_step, init_score_error
        if last_probe_step == global_step:
            return
        val, arrays = evaluate(model, val_loader, "stsb", device)
        value = val["full"]["spearman"]
        if global_step == 0:
            init_score_error = float(
                np.max(np.abs(
                    arrays["score_full"] - ref_arrays["score_full"]
                ))
            )
            print(f"[initial identity] max_abs_prediction_error="
                  f"{init_score_error:.3e}", flush=True)
            if init_score_error > 2e-4:
                raise RuntimeError(
                    "Initial B2 predictions differ from trained Mean. "
                    f"Max error={init_score_error:.3e}"
                )

        record = {
            "epoch": epoch, "global_step": global_step,
            "phase": phase, "last_minibatch_train_mse": last_batch_loss,
            "frozen_reference_mean": ref_eval["full"],
            "full_flag_b2": val["full"],
            "same_encoder_mean_with_current_head": val["same_encoder_mean"],
            "flag_minus_original_mean_spearman": value - ref_sp,
            "mean_abs_pair_cos_change": val["mean_abs_pair_cos_delta"],
            "mean_sentence_cos_to_original_mean":
                val["mean_sentence_cos_to_same_encoder_mean"],
            "head_scale": val["head_scale"], "head_bias": val["head_bias"],
        }
        history.append(record)
        with (out / "trajectory.jsonl").open("w", encoding="utf-8") as f:
            for item in history:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        np.savez_compressed(
            out / f"validation_pairs_step_{global_step}.npz", **arrays,
        )
        if np.isfinite(value) and value > best_metric:
            best_metric, best_step = value, global_step
            torch.save(
                pool_and_head_snapshot(model, checkpoint, epoch, global_step),
                out / "best_pool_head.pt",
            )
        print(
            f"[frozen B2][ep={epoch} step={global_step} {phase}] "
            f"Spearman={value:.6f}, "
            f"Mean_ref={ref_sp:.6f}, "
            f"delta={value-ref_sp:+.6f}, "
            f"pair_cos_diff={val['mean_abs_pair_cos_delta']:.6f}, "
            f"val_mse={val['full']['mse']:.6f}, "
            f"last_batch_mse={last_batch_loss}",
            flush=True,
        )
        last_probe_step = global_step
        model.train()

    snapshot(0, "initial")
    for epoch in range(1, args.flag_epochs + 1):
        model.train()
        last_batch_loss = None
        for tok_a, tok_b, y in train_loader:
            a = {key: value.to(device) for key, value in tok_a.items()}
            b = {key: value.to(device) for key, value in tok_b.items()}
            target = y.to(device) / 5.0
            opt.zero_grad(set_to_none=True)
            predicted, _ = model(a, b)
            loss = F.mse_loss(predicted, target)
            loss.backward()
            opt.step()
            global_step += 1
            last_batch_loss = float(loss.detach().cpu())
            if global_step in args.probe_steps:
                snapshot(epoch, "step_probe", last_batch_loss)
        snapshot(epoch, "epoch_end", last_batch_loss)

    torch.save(
        pool_and_head_snapshot(model, checkpoint, args.flag_epochs, global_step),
        out / "last_pool_head.pt",
    )
    save_json(out / "summary.json", {
        "experiment": "FLaG_B2 starts from trained STSB Mean, RoBERTa frozen",
        "seed": args.seed,
        "mean_reference_checkpoint": str(checkpoint),
        "mean_reference_origin": origin,
        "mean_reference_validation": ref_eval["full"],
        "b2_init_max_abs_pool_difference": pool_initial_max_error,
        "b2_init_max_abs_prediction_difference": init_score_error,
        "best_b2_step": best_step,
        "best_b2_val_spearman": best_metric,
        "best_b2_minus_mean_spearman": best_metric - ref_sp,
        "final_b2_val_spearman": history[-1]["full_flag_b2"]["spearman"],
        "final_b2_minus_mean_spearman":
            history[-1]["full_flag_b2"]["spearman"] - ref_sp,
        "steps": global_step,
        "flag_epochs": args.flag_epochs,
        "mean_reference_epochs_if_refitted": args.reference_epochs,
        "frozen_roberta": True,
        "flag_only_trainable_modules": ["FLaG_B2.pool", "log_scale", "bias"],
        "flag_pool": "masked_mean", "post_pool_norm": False,
        "gate_init": 1.0, "projection_init": "identity", "dropout": 0.0,
        "optimizer": "Adam", "lr": 1e-3, "weight_decay": 0.0,
        "checkpoint_selection": "validation Spearman, including step 0",
        "dataset": "STSB official train and validation; test not evaluated",
        "n_train": len(train_data), "n_validation": len(val_data),
    })
    print(f"\nDone. Share these two files:\n"
          f"  {out / 'summary.json'}\n"
          f"  {out / 'trajectory.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
