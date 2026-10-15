#!/usr/bin/env python3
"""Read-only sentence-pair ranking diagnostics for Mean-equivalent FLaG.

Uses val_pairs_epoch_*.npz from explore_mean_equiv_trajectory.py.
Does not train, evaluate test data, edit checkpoints, or change reports.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {name: f[name].astype(np.float64) for name in f.files}


def metrics_sprint(y, score):
    pos = score[y == 1]
    neg = score[y == 0]
    npos = len(pos)
    if not npos or not len(neg):
        raise ValueError("Need both positive and negative Sprint examples")
    order = np.argsort(-score, kind="mergesort")
    pos_ranks = np.flatnonzero(y[order] == 1) + 1
    def percentile(v, q):
        return float(np.percentile(v, q))
    return {
        "ap": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "positives": npos,
        "negatives": len(neg),
        "positive_cos_p10": percentile(pos, 10),
        "positive_cos_median": percentile(pos, 50),
        "positive_cos_p90": percentile(pos, 90),
        "negative_cos_p10": percentile(neg, 10),
        "negative_cos_median": percentile(neg, 50),
        "negative_cos_p90": percentile(neg, 90),
        "median_positive_rank": percentile(pos_ranks, 50),
        "positives_in_top_50": int(np.sum(y[order[:50]])),
        "positives_in_top_100": int(np.sum(y[order[:100]])),
        "positives_in_top_npositives": int(np.sum(y[order[:npos]])),
        "positive_recall_in_top_npositives": float(np.mean(y[order[:npos]])),
    }


def sprint_corrected_broken(y, old, new):
    # Count positive-negative pairs ranked correctly by new but not by old,
    # and vice versa. n_pos * n_neg is approx 1.19M for this Sprint validation.
    positive = y == 1
    before = old[positive, None] > old[None, ~positive]
    after = new[positive, None] > new[None, ~positive]
    correct = int(np.count_nonzero(~before & after))
    broken = int(np.count_nonzero(before & ~after))
    return {
        "positive_negative_comparisons": int(before.size),
        "corrected_vs_same_encoder_mean": correct,
        "broken_vs_same_encoder_mean": broken,
        "net_corrected_minus_broken": correct - broken,
        "corrected_percent": float(100 * correct / before.size),
        "broken_percent": float(100 * broken / before.size),
        "note": "Strict >; ties counted as incorrect. Descriptive pairs are not independent.",
    }


def stsb_corrected_broken(y, old, new):
    # Gold ties do not have a required ranking, so omit them.
    i, j = np.triu_indices(len(y), k=1)
    desired = np.sign(y[i] - y[j])
    valid = desired != 0
    i, j, desired = i[valid], j[valid], desired[valid]
    before = np.sign(old[i] - old[j]) == desired
    after = np.sign(new[i] - new[j]) == desired
    corrected = int(np.count_nonzero(~before & after))
    broken = int(np.count_nonzero(before & ~after))
    return {
        "comparable_distinct_label_pairs": int(len(i)),
        "corrected": corrected,
        "broken": broken,
        "net_corrected_minus_broken": corrected - broken,
        "corrected_percent": float(100 * corrected / len(i)),
        "broken_percent": float(100 * broken / len(i)),
        "note": "Gold ties omitted; prediction ties treated as incorrect; pairs not independent.",
    }


def one_seed(base, task, seed):
    root = base / task
    flag_folder = root / "flag_b2" / f"seed_{seed}"
    mean_folder = root / "mean" / f"seed_{seed}"
    files = sorted(
        flag_folder.glob("val_pairs_epoch_*.npz"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not files:
        raise FileNotFoundError(f"Missing FLaG pairs in: {flag_folder}")
    output = []
    for file in files:
        epoch = int(file.stem.split("_")[-1])
        f = load_npz(file)
        y = f["labels"]
        b2 = f["cos_full"]
        within_mean = f["cos_same_encoder_mean"]
        mean_file = mean_folder / file.name
        if not mean_file.is_file():
            raise FileNotFoundError(f"Missing paired raw-Mean file: {mean_file}")
        independent = load_npz(mean_file)
        if not np.array_equal(y, independent["labels"]):
            raise ValueError(f"Validation-pair order mismatch at {task} epoch {epoch}")
        external_mean = independent["cos_full"]
        row = {"epoch": epoch}
        if task == "sprint":
            if not np.all((y == 0) | (y == 1)):
                raise ValueError("Sprint labels must be 0/1")
            row["full"] = metrics_sprint(y, b2)
            row["same_encoder_mean"] = metrics_sprint(y, within_mean)
            row["independent_mean"] = metrics_sprint(y, external_mean)
            row["pairwise_full_vs_same_encoder_mean"] = (
                sprint_corrected_broken(y, within_mean, b2)
            )
        else:
            row["spearman"] = {
                "full": float(spearmanr(y, b2).statistic),
                "same_encoder_mean": float(spearmanr(y, within_mean).statistic),
                "independent_mean": float(spearmanr(y, external_mean).statistic),
            }
            row["full_vs_same_encoder_mean"] = (
                stsb_corrected_broken(y, within_mean, b2)
            )
            row["full_vs_independent_mean"] = (
                stsb_corrected_broken(y, external_mean, b2)
            )
        output.append(row)

    report = root / f"seed_{seed}_pairwise_diagnostics.json"
    report.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{task.upper()} seed {seed} | {len(output)} checkpoints")
    if task == "sprint":
        print("Epoch | AP Full | AUC Full | AP Mean | Pos@Top100 "
              "| Median Pos Rank | Corrected/Broken pos-neg pairs")
        for row in output:
            a = row["full"]
            b = row["same_encoder_mean"]
            c = row["pairwise_full_vs_same_encoder_mean"]
            print(
                f"{row['epoch']:>5} | {a['ap']:.4f}  | {a['roc_auc']:.4f}   "
                f"| {b['ap']:.4f}  | {a['positives_in_top_100']:>3} "
                f"| {a['median_positive_rank']:>8.0f} "
                f"| {c['corrected_vs_same_encoder_mean']:,}/"
                f"{c['broken_vs_same_encoder_mean']:,}"
            )
        print("\nPositive/negative cosine medians (B2, useful epochs):")
        for row in output:
            if row["epoch"] in (0, 1, 2, 3, 5, 7, 10):
                a = row["full"]
                print(
                    f" epoch {row['epoch']:>2}: positive={a['positive_cos_median']:+.4f},"
                    f" negative={a['negative_cos_median']:+.4f},"
                    f" positive-in-top-119={a['positives_in_top_npositives']}"
                )
    else:
        print("Epoch | Spearman Full | Same-encoder Mean | Independent Mean "
              "| Fixed encoder corrected/broken | vs independent corrected/broken")
        for row in output:
            sp = row["spearman"]
            own = row["full_vs_same_encoder_mean"]
            ind = row["full_vs_independent_mean"]
            print(
                f"{row['epoch']:>5} | {sp['full']:.6f} "
                f"| {sp['same_encoder_mean']:.6f} "
                f"| {sp['independent_mean']:.6f} "
                f"| {own['corrected']:,}/{own['broken']:,} "
                f"| {ind['corrected']:,}/{ind['broken']:,}"
            )
    print(f"Full detail: {report}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_dir", default="outputs/text/mean_equiv_exploration")
    p.add_argument("--tasks", nargs="+", default=["stsb", "sprint"],
                   choices=["stsb", "sprint"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = p.parse_args()
    for task in args.tasks:
        for seed in args.seeds:
            one_seed(Path(args.base_dir), task, seed)


if __name__ == "__main__":
    main()
