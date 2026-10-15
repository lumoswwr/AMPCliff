#!/usr/bin/env python3
"""Read-only STSB check: does B2 depend on dynamic padding / batch size?

Reuses a frozen, fine-tuned Mean RoBERTa and the FINAL trained FLaG_B2
pooling/head checkpoint. No optimizer, no fitting, no test set.

Two separate checks:
 (1) SAME cached RoBERTa hidden states, with only extra trailing zero-padding
     added before passing them to Mean and FLaG_B2 poolers. This isolates
     padding sensitivity in the pooling operation.
 (2) End-to-end evaluation of the SAME validation pairs under dynamic
     batch sizes 1/32/64 and fixed length 128. This measures combined
     batch-shape / padding effects. Mean acts as a negative control.
Do not confuse batch size with padded sequence length.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from AMPCliff.factory.pooling.llm_pooling_dropin import masked_mean_pooling
from explore_mean_equiv_trajectory import evaluate, get_splits
from train_author_text_protocol import PairDataset, PairCollator
from train_stsb_frozen_mean_start import build_frozen_b2


def numpy_stats(v):
    a = np.asarray(v, dtype=np.float64)
    return {
        "mean": float(a.mean()), "p95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
    }


class FixedLengthCollator:
    def __init__(self, tokenizer, fixed=None):
        self.inner = PairCollator(tokenizer, max_length=128, label_dtype=torch.float32)
        self.fixed = fixed
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer has no pad token")

    def __call__(self, pairs):
        a, b, y = self.inner(pairs)
        if self.fixed is None:
            return a, b, y
        length = a["input_ids"].shape[1]
        if length > self.fixed:
            raise ValueError(f"Got token length {length} > fixed {self.fixed}")
        n = self.fixed - length
        if n == 0:
            return a, b, y
        for tok in (a, b):
            for key, tensor in tok.items():
                value = self.pad_id if key == "input_ids" else 0
                tok[key] = F.pad(tensor, (0, n), value=value)
        return a, b, y


def loader(dataset, tokenizer, batch_size, fixed=None):
    return DataLoader(
        PairDataset(dataset, "stsb"), batch_size=batch_size,
        shuffle=False, num_workers=0,
        collate_fn=FixedLengthCollator(tokenizer, fixed),
    )


def compare_arrays(base, current):
    if not np.array_equal(base["labels"], current["labels"]):
        raise RuntimeError("Validation example ordering or labels changed")
    report = {}
    for name in ("cos_full", "cos_same_encoder_mean",
                 "score_full", "score_same_encoder_mean"):
        report[name] = numpy_stats(np.abs(
            current[name].astype(np.float64) -
            base[name].astype(np.float64)
        ))
    return report


def isolating_pool_padding(model, dataset, tokenizer, device, max_examples):
    """Reuse exactly the same encoder hidden states, then change only pad count."""
    model.eval()
    examples = []
    for row in dataset:
        for sentence_key in ("sentence1", "sentence2"):
            text = row[sentence_key]
            enc = tokenizer(text, truncation=True, max_length=128)
            length = len(enc["input_ids"])
            if 4 <= length <= 32:
                examples.append((text, length))
                if len(examples) >= max_examples:
                    break
        if len(examples) >= max_examples:
            break
    if not examples:
        raise RuntimeError("No sentence <=32 tokens available for isolated check")

    targets = (32, 64, 128)
    out = []
    with torch.no_grad():
        for idx, (sentence, length) in enumerate(examples):
            tok = tokenizer(sentence, return_tensors="pt",
                            truncation=True, max_length=128)
            tok = {k: v.to(device) for k, v in tok.items()}
            hidden = model.encoder(**tok).last_hidden_state
            clean_mask = tok["attention_mask"]
            base_mean = F.normalize(
                masked_mean_pooling(hidden, clean_mask), p=2, dim=-1,
            )
            base_flag = F.normalize(
                model.pool(hidden, attention_mask=clean_mask), p=2, dim=-1,
            )
            results = []
            for target in targets:
                extra = target - hidden.shape[1]
                if extra < 0:
                    continue
                # Important: no RoBERTa forward on these padded inputs.
                # Identical valid token hidden states, zero vectors in pad slots.
                padded_hidden = F.pad(hidden, (0, 0, 0, extra), value=0.0)
                padded_mask = F.pad(clean_mask, (0, extra), value=0)
                mean = F.normalize(
                    masked_mean_pooling(padded_hidden, padded_mask),
                    p=2, dim=-1,
                )
                flag = F.normalize(
                    model.pool(padded_hidden, attention_mask=padded_mask),
                    p=2, dim=-1,
                )
                mean_cos = float(F.cosine_similarity(base_mean, mean).item())
                flag_cos = float(F.cosine_similarity(base_flag, flag).item())
                results.append({
                    "target_length": target,
                    "mean_L2_diff": float((mean - base_mean).norm().item()),
                    "flag_L2_diff": float((flag - base_flag).norm().item()),
                    "mean_cos_to_unpadded": mean_cos,
                    "flag_cos_to_unpadded": flag_cos,
                })
            out.append({
                "example_index": idx,
                "unpadded_length": length,
                "length_changes": results,
            })
    aggregates = {}
    for target in targets:
        arr = [r for row in out for r in row["length_changes"]
               if r["target_length"] == target]
        aggregates[str(target)] = {
            "mean_L2": numpy_stats([v["mean_L2_diff"] for v in arr]),
            "flag_L2": numpy_stats([v["flag_L2_diff"] for v in arr]),
            "n_sentences": len(arr),
        }
    return {
        "n_sentence_examples": len(examples),
        "summary_by_target_length": aggregates,
        "per_sentence": out,
        "method": "Same frozen RoBERTa hidden states, padded with literal zeros; "
                  "only pooling sees different FFT input lengths.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_path",
                   default="/home/data/home/wwr_lumos/models/roberta-base")
    p.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    p.add_argument("--pilot_dir", default="outputs/text/stsb_frozen_trained_mean")
    p.add_argument("--n_pairs", type=int, default=256,
                   help="Validation subset for quick diagnostic; 0=all 1500.")
    p.add_argument("--n_isolated_sentences", type=int, default=12)
    args = p.parse_args()

    out = Path(args.pilot_dir) / f"seed_{args.seed}"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if summary.get("seed") != args.seed or not summary.get("frozen_roberta"):
        raise ValueError("Requires matching seed frozen-Mean pilot")
    refpath = Path(summary["mean_reference_checkpoint"])
    last = torch.load(out / "last_pool_head.pt",
                      map_location="cpu", weights_only=True)
    if not refpath.is_file() or str(refpath) != str(last["reference_checkpoint"]):
        raise ValueError("Missing or mismatched Mean reference checkpoint")
    if int(last["step"]) != int(summary["steps"]):
        raise ValueError("Last B2 checkpoint does not match pilot summary")
    ref_state = torch.load(refpath, map_location="cpu", weights_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True,
    )
    train, val = get_splits(
        "stsb", args.stsb_data_path, "", args.seed, 0, 0
    )
    if len(train) != summary["n_train"] or len(val) != summary["n_validation"]:
        raise RuntimeError("Dataset split size differs from original pilot")
    if args.n_pairs:
        val = val.select(range(min(args.n_pairs, len(val))))
    model = build_frozen_b2(args, device, ref_state)
    model.pool.load_state_dict(last["pool"], strict=True)
    with torch.no_grad():
        model.log_scale.copy_(last["log_scale"].to(device))
        model.bias.copy_(last["bias"].to(device))
    model.eval()

    isolated = isolating_pool_padding(
        model, val, tokenizer, device, args.n_isolated_sentences
    )
    conditions = [
        ("dynamic_batch_64", 64, None),
        ("dynamic_batch_32", 32, None),
        ("dynamic_batch_1", 1, None),
        ("fixed_128_batch_64", 64, 128),
    ]
    evaluations = {}
    baseline_pairs = None
    for name, batch_size, fixed in conditions:
        print(f"[Eval] {name} n_pairs={len(val)}", flush=True)
        result, arrays = evaluate(
            model, loader(val, tokenizer, batch_size, fixed),
            "stsb", device
        )
        record = {
            "batch_size": batch_size,
            "padding": "dynamic" if fixed is None else f"fixed_{fixed}",
            "mean": result["same_encoder_mean"],
            "flag_b2": result["full"],
            "shape_stats": {
                "mean_abs_pair_cos_delta": result["mean_abs_pair_cos_delta"],
            },
        }
        if baseline_pairs is None:
            baseline_pairs = arrays
        record["delta_vs_dynamic_batch_64"] = compare_arrays(
            baseline_pairs, arrays
        )
        evaluations[name] = record
        print(
            f"  Mean Spearman={record['mean']['spearman']:.6f} | "
            f"FLaG Spearman={record['flag_b2']['spearman']:.6f} | "
            f"FLaG mean abs score diff vs baseline="
            f"{record['delta_vs_dynamic_batch_64']['score_full']['mean']:.6g}",
            flush=True
        )

    # The original pilot used full validation, dynamic batch64.
    if len(val) == summary["n_validation"]:
        expected = float(summary["final_b2_val_spearman"])
        actual = evaluations["dynamic_batch_64"]["flag_b2"]["spearman"]
        if abs(expected - actual) > 5e-4:
            raise RuntimeError(
                "Baseline does not reproduce prior validation score: "
                f"{actual:.6f} vs {expected:.6f}"
            )
    output = {
        "description": "Padding/batch sensitivity of trained B2 vs Mean",
        "n_pairs": len(val),
        "full_validation": len(val) == summary["n_validation"],
        "checkpoint_step": int(last["step"]),
        "model_is_frozen": True,
        "fixed_length": 128,
        "no_training": True,
        "nonscientific_conclusion_before_running": None,
        "isolated_pooling": isolated,
        "end_to_end": evaluations,
        "notes": [
            "Mean negative control; same paired data order and weights.",
            "Isolated test reuses exactly the same encoder hidden states.",
            "Batch-size test includes numerical shape effects in RoBERTa.",
            "Spearman on n_pairs < 1500 is subset-specific, not official.",
        ],
    }
    destination = out / "padding_sensitivity.json"
    destination.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    print("\nISOLATED POOLING: mean/L2 of single-sentence vector changes "
          "vs unpadded hidden states", flush=True)
    for length, v in isolated["summary_by_target_length"].items():
        print(f"  FFT length={length}: Mean L2={v['mean_L2']['mean']:.6g}, "
              f"FLaG L2={v['flag_L2']['mean']:.6g}")
    print(f"\nSaved: {destination}", flush=True)


if __name__ == "__main__":
    main()
