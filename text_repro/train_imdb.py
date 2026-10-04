import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from train_sprint import SprintEncoder


class IMDBDataset(Dataset):
    def __init__(self, split):
        self.data = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return {"text": x["text"], "label": int(x["label"])}


class IMDBCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        tokens = self.tokenizer(
            [x["text"] for x in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        return tokens, labels


class IMDBClassifier(nn.Module):
    def __init__(self, model_path, pooling, **kwargs):
        super().__init__()
        self.encoder = SprintEncoder(
            model_path=model_path,
            pooling=pooling,
            **kwargs,
        )
        hidden = self.encoder.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, tokens):
        hidden = self.encoder.backbone(**tokens).last_hidden_state
        z = self.encoder.pool(hidden, tokens["attention_mask"])
        return self.classifier(z)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for tokens, labels in loader:
            tokens = {k: v.to(device) for k, v in tokens.items()}
            logits = model(tokens)
            pred = logits.argmax(dim=-1).cpu().numpy()
            ys.extend(labels.numpy())
            ps.extend(pred)
    return accuracy_score(ys, ps), f1_score(ys, ps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/text/imdb")
    parser.add_argument("--model_path", default="/home/data/home/wwr_lumos/models/roberta-base")
    parser.add_argument("--pooling", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--output_dir", default="outputs/text/imdb/experiments")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = load_from_disk(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    train_loader = DataLoader(
        IMDBDataset(raw["train"]),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=IMDBCollator(tokenizer, args.max_length),
    )
    test_loader = DataLoader(
        IMDBDataset(raw["test"]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=IMDBCollator(tokenizer, args.max_length),
    )

    model = IMDBClassifier(
        args.model_path,
        args.pooling,
        finetune_backbone=False,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    loss_fn = nn.CrossEntropyLoss()

    best = -1
    run_dir = Path(args.output_dir) / args.experiment_name / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for tokens, labels in train_loader:
            tokens = {k: v.to(device) for k, v in tokens.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(tokens), labels)
            loss.backward()
            optimizer.step()

        acc, f1 = evaluate(model, test_loader, device)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    acc, f1 = evaluate(model, test_loader, device)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({
            "pooling": args.pooling,
            "test_accuracy": acc,
            "test_f1": f1,
            "seed": args.seed,
        }, f, indent=2)


if __name__ == "__main__":
    main()
