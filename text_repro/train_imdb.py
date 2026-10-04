import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm

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
    def __init__(self, model_path, pooling):
        super().__init__()
        self.encoder = SprintEncoder(
            model_path=model_path,
            pooling=pooling,
            finetune_backbone=False,
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
            pred = model(tokens).argmax(dim=-1).cpu().numpy()
            ys.extend(labels.numpy())
            ps.extend(pred)
    return accuracy_score(ys, ps), f1_score(ys, ps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/text/imdb')
    parser.add_argument('--model_path', default='/home/data/home/wwr_lumos/models/roberta-base')
    parser.add_argument('--pooling', required=True)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--output_dir', default='outputs/text/imdb/experiments')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw = load_from_disk(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    full_train = IMDBDataset(raw['train'])
    val_size = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    collator = IMDBCollator(tokenizer, args.max_length)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(IMDBDataset(raw['test']), batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = IMDBClassifier(args.model_path, args.pooling).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    loss_fn = nn.CrossEntropyLoss()

    run_dir = Path(args.output_dir) / args.experiment_name / f'seed_{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val = -1
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for tokens, labels in bar:
            tokens = {k: v.to(device) for k, v in tokens.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(tokens), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            bar.set_postfix(loss=f'{loss.item():.4f}')

        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}: train_loss={total_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}')

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), run_dir / 'best_model.pt')

    model.load_state_dict(torch.load(run_dir / 'best_model.pt', map_location=device))
    test_acc, test_f1 = evaluate(model, test_loader, device)

    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump({
            'pooling': args.pooling,
            'best_epoch': best_epoch,
            'val_accuracy': best_val,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'seed': args.seed,
        }, f, indent=2)


if __name__ == '__main__':
    main()
