import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer

from text_repro.train_sprint import (
    SprintDataset,
    SprintCollator,
    SprintPairClassifier,
    evaluate,
)


def best_accuracy_threshold(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # Start above max score: predict all negative.
    correct = int((labels == 0).sum())
    best_acc = correct / len(labels)
    best_threshold = np.nextafter(sorted_scores[-1], np.inf)

    # Move threshold downward. Each crossed score becomes positive.
    for i in range(len(sorted_scores) - 1, -1, -1):
        if sorted_labels[i] == 1:
            correct += 1
        else:
            correct -= 1

        if i == 0 or sorted_scores[i - 1] < sorted_scores[i]:
            acc = correct / len(labels)
            if acc > best_acc:
                best_acc = acc
                if i == 0:
                    best_threshold = np.nextafter(sorted_scores[0], -np.inf)
                else:
                    best_threshold = (
                        sorted_scores[i - 1] + sorted_scores[i]
                    ) / 2.0

    return float(best_acc), float(best_threshold)


def best_f1_threshold(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    tp = 0
    fp = 0
    total_pos = int(labels.sum())

    best = {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "threshold": np.nextafter(sorted_scores[0], np.inf),
    }

    i = 0
    n = len(sorted_scores)

    while i < n:
        score = sorted_scores[i]
        j = i

        while j < n and sorted_scores[j] == score:
            if sorted_labels[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1

        fn = total_pos - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / total_pos if total_pos else 0.0

        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        if f1 > best["f1"]:
            if j < n:
                threshold = (score + sorted_scores[j]) / 2.0
            else:
                threshold = np.nextafter(score, -np.inf)

            best = {
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "threshold": float(threshold),
            }

        i = j

    return best


def metrics_at_threshold(scores, labels, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    preds = (scores >= threshold).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(
            precision_score(labels, preds, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, preds, zero_division=0)
        ),
        "predicted_positive_rate": float(preds.mean()),
        "predicted_positive_count": int(preds.sum()),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir",
        required=True,
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device:", device)
    print("run_dir:", run_dir)

    raw = load_from_disk(cfg["data_path"])

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_path"],
        local_files_only=True,
    )

    val_ds = SprintDataset(raw["validation"])

    collator = SprintCollator(
        tokenizer,
        max_length=cfg["max_length"],
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model = SprintPairClassifier(
        model_path=cfg["model_path"],
        pooling=cfg["pooling"],
        stft_win_length=cfg.get(
            "stft_win_length",
            16,
        ),
        stft_hop_length=cfg.get(
            "stft_hop_length",
            16,
        ),
        stft_window_type=cfg.get(
            "stft_window_type",
            "rect",
        ),
        stft_center=cfg.get(
            "stft_center",
            False,
        ),
    ).to(device)

    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        default_metrics,
        logits,
        labels,
        cosines,
    ) = evaluate(
        model,
        val_loader,
        device,
    )

    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    cosines = np.asarray(cosines, dtype=np.float64)

    best_acc, best_acc_threshold = best_accuracy_threshold(
        logits,
        labels,
    )

    best_f1 = best_f1_threshold(
        logits,
        labels,
    )

    scale = float(
        model.head.positive_scale()
        .detach()
        .cpu()
    )

    bias = float(
        model.head.bias
        .detach()
        .cpu()
    )

    acc_cos_threshold = (
        best_acc_threshold - bias
    ) / scale

    f1_cos_threshold = (
        best_f1["threshold"] - bias
    ) / scale

    result = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "average_precision": float(
            average_precision_score(labels, logits)
        ),
        "cosine_average_precision": float(
            average_precision_score(labels, cosines)
        ),
        "head_scale": scale,
        "head_bias": bias,
        "default_logit_zero": default_metrics,
        "best_accuracy": {
            "value": best_acc,
            "logit_threshold": best_acc_threshold,
            "cosine_threshold": float(acc_cos_threshold),
            **metrics_at_threshold(
                logits,
                labels,
                best_acc_threshold,
            ),
        },
        "best_f1": {
            "logit_threshold": best_f1["threshold"],
            "cosine_threshold": float(f1_cos_threshold),
            **metrics_at_threshold(
                logits,
                labels,
                best_f1["threshold"],
            ),
        },
    }

    print("\n================================")
    print("VALIDATION THRESHOLD DIAGNOSTIC")
    print("================================")

    print("checkpoint epoch:", result["checkpoint_epoch"])
    print("AP:", result["average_precision"])
    print("cosAP:", result["cosine_average_precision"])
    print("head scale:", scale)
    print("head bias:", bias)

    print("\n--- default logit >= 0 ---")
    for k, v in result["default_logit_zero"].items():
        print(f"{k}: {v}")

    print("\n--- best validation accuracy threshold ---")
    for k, v in result["best_accuracy"].items():
        print(f"{k}: {v}")

    print("\n--- best validation F1 threshold ---")
    for k, v in result["best_f1"].items():
        print(f"{k}: {v}")

    out = run_dir / "validation_threshold_diagnostic.json"

    with open(out, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print("\nsaved:", out)


if __name__ == "__main__":
    main()
