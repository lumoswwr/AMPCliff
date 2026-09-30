#!/usr/bin/env python3
"""Inspect local Sprint and optionally STSB using the experiment tokenizer.

From the AMPCliff repo root:
  python text_repro/inspect_sprint_length.py \
    --sprint data/text/sprintduplicatequestions_adaptation \
    --stsb data/text/stsbenchmark \
    --tokenizer /home/data/home/wwr_lumos/models/roberta-base \
    | tee sprint_length_report.txt

Token length includes special tokens, excludes batch padding, and is measured
before max_length=128 truncation. For E12 (win=16, hop=16, center=False),
actual token length <=16 gives exactly one valid STFT frame.
"""
import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer


def describe(values, title):
    arr = np.asarray(values, dtype=np.int64)
    if not len(arr):
        print(f"  {title}: empty")
        return
    print(f"  {title:<17} mean={arr.mean():5.1f}  p50={np.median(arr):4.0f}  "
          f"p90={np.percentile(arr,90):4.0f}  p95={np.percentile(arr,95):4.0f}  "
          f"max={arr.max():4d}")


def load_dataset(path, name):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{name} path not found: {p.resolve()}")
    data = load_from_disk(str(p))
    for split in data:
        if any(col not in data[split].column_names for col in ("sentence1", "sentence2")):
            raise SystemExit(f"{name}/{split} missing sentence columns")
    return data


def build_length_lookup(datasets, tokenizer):
    # Tokenize distinct texts once; final percentages still count repeated pairs.
    texts = set()
    for ds in datasets:
        for split in ds:
            texts.update(ds[split]["sentence1"])
            texts.update(ds[split]["sentence2"])
    texts = sorted(texts)
    print(f"\nTokenizing {len(texts):,} distinct texts using LOCAL tokenizer.")
    lookup = {}
    for start in range(0, len(texts), 256):
        batch = texts[start:start+256]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False,
                            padding=False, return_length=True, verbose=False)
        lookup.update(zip(batch, encoded["length"]))
    return lookup


def stats_for_split(name, split, lengths, kind):
    first = split["sentence1"]
    second = split["sentence2"]
    a = np.asarray([lengths[x] for x in first], dtype=np.int64)
    b = np.asarray([lengths[x] for x in second], dtype=np.int64)
    print(f"\n--- {name} ({len(first):,} pairs) ---")
    print(f"  distinct sentence1={len(set(first)):,}, sentence2={len(set(second)):,}")
    describe([len(x) for x in first], "Q1 chars")
    describe([len(x) for x in second], "Q2 chars")
    describe([len(x.split()) for x in first], "Q1 words")
    describe([len(x.split()) for x in second], "Q2 words")
    describe(a, "Q1 RoBERTa tokens")
    describe(b, "Q2 RoBERTa tokens")
    describe(np.concatenate((a, b)), "All RoBERTa toks")
    print("  Token lengths include <s> and </s> and exclude batch padding:")
    for cutoff in (8, 16, 32, 64, 128):
        print(f"    <= {cutoff:3d}: Q1={np.mean(a<=cutoff):6.2%} "
              f"Q2={np.mean(b<=cutoff):6.2%} "
              f"both-in-pair={np.mean((a<=cutoff)&(b<=cutoff)):6.2%}")
    print(f"    >128 (truncated during training): "
          f"Q1={np.mean(a>128):.2%}, Q2={np.mean(b>128):.2%}")
    if kind == "sprint":
        labs = np.asarray([int(x) for x in split["label"]])
        counts = Counter(labs.tolist())
        print(f"  labels: {dict(sorted(counts.items()))}; positive={np.mean(labs==1):.3%}")
        for lab in (0, 1):
            mask = labs == lab
            if not mask.any():
                continue
            print(f"  label={lab}, pairs={mask.sum():,}: "
                  f"Q1-token p50={np.median(a[mask]):.0f}, "
                  f"Q2-token p50={np.median(b[mask]):.0f}, "
                  f"both<=16={np.mean((a[mask]<=16)&(b[mask]<=16)):.2%}")


def show_sprint_examples(split, lookup):
    print("\n=== Sprint examples (official test) ===")
    rng = random.Random(42)
    first, second = split["sentence1"], split["sentence2"]
    labels = [int(z) for z in split["label"]]
    a = np.asarray([lookup[x] for x in first])
    b = np.asarray([lookup[x] for x in second])
    for lab in (1, 0):
        ids = [i for i, z in enumerate(labels) if z == lab]
        if not ids:
            continue
        targets = [
            ("short, both<=16", [i for i in ids if a[i]<=16 and b[i]<=16]),
            ("longer, either>32", [i for i in ids if a[i]>32 or b[i]>32]),
            ("typical", sorted(ids, key=lambda i:
              abs(a[i]-np.median(a[ids])) + abs(b[i]-np.median(b[ids]))))
        ]
        for tag, pool in targets:
            chosen = pool[:2] if tag == "typical" else rng.sample(pool, min(2, len(pool)))
            print(f"\n  Label {lab} / {tag}: {len(pool):,} candidates")
            for i in chosen:
                print(f"    Q1 [{a[i]:3d} tokens]: {first[i]}")
                print(f"    Q2 [{b[i]:3d} tokens]: {second[i]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sprint", required=True, help="local Sprint DatasetDict")
    parser.add_argument("--stsb", help="optional local STSB DatasetDict")
    parser.add_argument("--tokenizer", required=True, help="local RoBERTa tokenizer")
    args = parser.parse_args()

    sprint = load_dataset(args.sprint, "Sprint")
    stsb = load_dataset(args.stsb, "STSB") if args.stsb else None
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                        use_fast=True)
    lookup = build_length_lookup([d for d in (sprint, stsb) if d is not None], tok)

    print("\nE12 has one valid frame when actual token length <=16.")
    print("Lengths are BEFORE truncation, with special tokens but no batch padding.")
    print("Percentages count sentence PAIRS, not distinct questions.")
    for split_name in ("train", "validation", "test"):
        if split_name in sprint:
            stats_for_split(f"SPRINT/{split_name}", sprint[split_name], lookup, "sprint")
    if "test" in sprint:
        show_sprint_examples(sprint["test"], lookup)
    if stsb is not None:
        print("\n=== STSB for comparison ===")
        for split_name in ("train", "validation", "test"):
            if split_name in stsb:
                stats_for_split(f"STSB/{split_name}", stsb[split_name], lookup, "stsb")


if __name__ == "__main__":
    main()
