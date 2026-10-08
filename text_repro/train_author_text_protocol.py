import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import concatenate_datasets, load_from_disk
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from AMPCliff.factory.pooling.flag_pooling import (
    FFTLatentAttentionGatePooling,
)
from AMPCliff.factory.pooling.static_flag_mechanism_pooling import (
    StaticFLaGMechanismPooling,
)


# =========================================================
# Reproducibility
# =========================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# Data helpers
# =========================================================

def split_90_10(dataset, seed: int):
    """
    Match the Pooling-img-text protocol:
    shuffle with split_seed=training.seed, then use ~90% train / ~10% val.
    """
    total = len(dataset)
    val_n = max(1, int(round(total * 0.1)))
    train_n = total - val_n

    rng = np.random.default_rng(seed)
    order = rng.permutation(total)

    train_idx = order[:train_n].tolist()
    val_idx = order[train_n:].tolist()

    return dataset.select(train_idx), dataset.select(val_idx)


class IMDBDataset(Dataset):
    def __init__(self, split):
        self.data = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return {
            "text": x["text"],
            "label": int(x["label"]),
        }


class IMDBCollator:
    def __init__(self, tokenizer, max_length: int):
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
        labels = torch.tensor(
            [x["label"] for x in batch],
            dtype=torch.long,
        )
        return tokens, labels


class PairDataset(Dataset):
    def __init__(self, split, task: str):
        self.data = split
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        if self.task == "stsb":
            return {
                "sentence1": x["sentence1"],
                "sentence2": x["sentence2"],
                "label": float(x["score"]),
            }

        if self.task == "sprint":
            return {
                "sentence1": x["sentence1"],
                "sentence2": x["sentence2"],
                "label": float(x["label"]),
            }

        raise ValueError(self.task)


class PairCollator:
    def __init__(self, tokenizer, max_length: int, label_dtype):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_dtype = label_dtype

    def __call__(self, batch):
        s1 = [x["sentence1"] for x in batch]
        s2 = [x["sentence2"] for x in batch]

        # Keep both sides on exactly the same padded sequence length.
        all_sentences = s1 + s2
        tokens = self.tokenizer(
            all_sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch_size = len(batch)
        tok1 = {k: v[:batch_size] for k, v in tokens.items()}
        tok2 = {k: v[batch_size:] for k, v in tokens.items()}

        labels = torch.tensor(
            [x["label"] for x in batch],
            dtype=self.label_dtype,
        )
        return tok1, tok2, labels


# =========================================================
# Pooling
# =========================================================

class MeanPooling(nn.Module):
    def forward(self, features, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(features.dtype)
        summed = (features * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom


def build_pooling(
    pooling: str,
    d_model: int,
    mean_anchor_beta_init: float,
    flag_time_pool: str = "max",
    flag_post_pool_norm: bool = True,
):
    if pooling == "mean":
        return MeanPooling()

    static_modes = {
        "StaticFLaG_ReIm_B2": "static_reim",
        "StaticFLaG_Diag_B2": "static_diag",
        "MeanProj_B2": "mean_project",
        "MeanProjRand": "mean_project_random",
    }
    if pooling in static_modes:
        if flag_time_pool != "mean" or flag_post_pool_norm:
            raise ValueError(
                "Matched static FLaG controls require --flag_time_pool mean "
                "and --disable_post_pool_norm."
            )
        return StaticFLaGMechanismPooling(
            d_model=d_model,
            mode=static_modes[pooling],
        )

    if pooling not in {
        "FLaG",
        "FLaG_A1",
        "FLaG_B1",
        "FLaG_A1Z",
        "FLaG_B2",
        "FLaG_AlignmentAnchor",
    }:
        raise ValueError(f"Unsupported pooling: {pooling}")

    # Defaults reproduce Kewei2023/Pooling-img-text. The optional time-pool,
    # post-norm and initialization controls are used only by matched mechanism
    # ablations; existing author-protocol runs remain unchanged by default.
    return FFTLatentAttentionGatePooling(
        d_model=d_model,
        num_latents=8,
        num_heads=4,
        dropout=0.0,
        time_pool=flag_time_pool,
        gate_residual=True,
        eps=1e-6,
        use_gate=True,
        use_latent=True,
        post_pool_norm=flag_post_pool_norm,
        window_type=None,
        fixed_fft_length=None,
        remove_dc=False,
        dc_only=False,
        mean_anchor_residual=(
            pooling == "FLaG_AlignmentAnchor"
        ),
        mean_anchor_beta_init=mean_anchor_beta_init,
        mean_alignment_anchor=(
            pooling == "FLaG_AlignmentAnchor"
        ),
        gate_parameterization=(
            "centered_sigmoid"
            if pooling in {"FLaG_B1", "FLaG_A1Z", "FLaG_B2"}
            else "residual_sigmoid"
        ),
        identity_time_out_proj=(
            pooling in {
                "FLaG_A1",
                "FLaG_B1",
                "FLaG_A1Z",
                "FLaG_B2",
            }
        ),
        zero_init_gate_output=(
            pooling in {"FLaG_A1Z", "FLaG_B2"}
        ),
    )


# =========================================================
# Models
# =========================================================

class IMDBModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        pooling: str,
        mean_anchor_beta_init: float,
        flag_time_pool: str = "max",
        flag_post_pool_norm: bool = True,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
        )
        d_model = self.encoder.config.hidden_size
        self.pool = build_pooling(
            pooling,
            d_model,
            mean_anchor_beta_init,
            flag_time_pool=flag_time_pool,
            flag_post_pool_norm=flag_post_pool_norm,
        )
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, tokens):
        hidden = self.encoder(**tokens).last_hidden_state
        pooled = self.pool(
            hidden,
            attention_mask=tokens["attention_mask"],
        )
        return self.classifier(pooled)


class SentencePairModel(nn.Module):
    """
    Match Pooling-img-text RoBERTaSentencePairModel:
      pooled -> L2 normalize -> cosine
      score = exp(log_scale) * cosine + bias
    """

    def __init__(
        self,
        model_path: str,
        pooling: str,
        freeze_backbone: bool,
        mean_anchor_beta_init: float,
        flag_time_pool: str = "max",
        flag_post_pool_norm: bool = True,
    ):
        super().__init__()
        self.freeze_backbone = bool(freeze_backbone)

        self.encoder = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
        )

        if self.freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        d_model = self.encoder.config.hidden_size
        self.pool = build_pooling(
            pooling,
            d_model,
            mean_anchor_beta_init,
            flag_time_pool=flag_time_pool,
            flag_post_pool_norm=flag_post_pool_norm,
        )

        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def _encode(self, tokens):
        if self.freeze_backbone:
            self.encoder.eval()
            with torch.no_grad():
                hidden = self.encoder(
                    **tokens
                ).last_hidden_state
        else:
            hidden = self.encoder(
                **tokens
            ).last_hidden_state

        pooled = self.pool(
            hidden,
            attention_mask=tokens["attention_mask"],
        )
        return F.normalize(
            pooled,
            p=2,
            dim=-1,
        )

    def forward(self, tok1, tok2):
        # The author's sentence-pair model encodes each side separately.
        z1 = self._encode(tok1)
        z2 = self._encode(tok2)

        cosine = (z1 * z2).sum(dim=-1)
        score = (
            torch.exp(self.log_scale)
            * cosine
            + self.bias
        )
        return score, cosine


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def eval_imdb(model, loader, device):
    model.eval()
    ys = []
    preds = []

    for tokens, labels in loader:
        tokens = {k: v.to(device) for k, v in tokens.items()}
        logits = model(tokens)
        pred = logits.argmax(dim=-1).cpu().numpy()

        ys.extend(labels.numpy().tolist())
        preds.extend(pred.tolist())

    return {
        "accuracy": float(accuracy_score(ys, preds)),
        "f1": float(f1_score(ys, preds, zero_division=0)),
    }


@torch.no_grad()
def eval_stsb(model, loader, device):
    model.eval()
    ys = []
    preds = []

    for tok1, tok2, labels in loader:
        tok1 = {k: v.to(device) for k, v in tok1.items()}
        tok2 = {k: v.to(device) for k, v in tok2.items()}

        score, _ = model(tok1, tok2)

        preds.extend(score.cpu().numpy().tolist())
        ys.extend(labels.numpy().tolist())

    sp = spearmanr(preds, ys).statistic
    pe = pearsonr(preds, ys).statistic

    if not np.isfinite(sp):
        sp = 0.0
    if not np.isfinite(pe):
        pe = 0.0

    return {
        "spearman": float(sp),
        "pearson": float(pe),
    }


@torch.no_grad()
def eval_sprint(model, loader, device):
    model.eval()
    ys = []
    logits_all = []

    for tok1, tok2, labels in loader:
        tok1 = {k: v.to(device) for k, v in tok1.items()}
        tok2 = {k: v.to(device) for k, v in tok2.items()}

        logits, _ = model(tok1, tok2)

        logits_all.extend(logits.cpu().numpy().tolist())
        ys.extend(labels.numpy().astype(np.int64).tolist())

    logits_np = np.asarray(logits_all, dtype=np.float64)
    labels_np = np.asarray(ys, dtype=np.int64)

    probs = 1.0 / (
        1.0 + np.exp(-np.clip(logits_np, -50.0, 50.0))
    )
    preds = (probs >= 0.5).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(labels_np, preds)),
        "average_precision": float(
            average_precision_score(labels_np, probs)
        ),
        "f1": float(
            f1_score(labels_np, preds, zero_division=0)
        ),
        "precision": float(
            precision_score(labels_np, preds, zero_division=0)
        ),
        "recall": float(
            recall_score(labels_np, preds, zero_division=0)
        ),
    }



# =========================================================
# Stage-wise FLaG angle audit
# =========================================================

def _masked_mean_tensor(features, attention_mask, eps=1e-6):
    mask = attention_mask.unsqueeze(-1).to(features.dtype)
    summed = (features * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=eps)
    return summed / denom


def _angle_deg(a, b):
    cosine = F.cosine_similarity(
        a,
        b,
        dim=-1,
        eps=1e-8,
    ).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def _stage_angle_batch(pool, hidden, attention_mask):
    required = [
        "_last_freq_tokens",
        "_last_enhanced_freq",
        "_last_time_tokens",
        "_last_pooled_pre_projection",
        "_last_pooled_for_projection",
        "_last_pooled_output",
    ]
    missing = [
        name for name in required
        if not hasattr(pool, name)
    ]
    if missing:
        raise RuntimeError(
            "Stage-angle audit requires FLaG stage tensors; "
            f"missing {missing}"
        )

    d_model = hidden.size(-1)
    mean_input = _masked_mean_tensor(
        hidden,
        attention_mask,
    )

    freq = pool._last_freq_tokens
    enhanced = pool._last_enhanced_freq

    # k=0 real coefficient is the spectral DC vector. The imaginary DC
    # coefficient is zero for real-valued input and remains zero under the
    # channel-wise gate, so comparing the real D-dimensional vectors is enough.
    dc_before = freq[:, 0, :d_model]
    dc_after = enhanced[:, 0, :d_model]

    time_mean = _masked_mean_tensor(
        pool._last_time_tokens,
        attention_mask,
    )
    pooled_pre = pool._last_pooled_pre_projection
    pooled_postnorm = pool._last_pooled_for_projection
    pooled_final = pool._last_pooled_output

    return {
        # Sanity check for the theoretical Mean <-> DC relation.
        "mean_to_dc_deg": _angle_deg(
            mean_input,
            dc_before,
        ),
        # Whole-spectrum change induced by the gate.
        "spectral_gate_deg": _angle_deg(
            freq.reshape(freq.size(0), -1),
            enhanced.reshape(enhanced.size(0), -1),
        ),
        # The key channel-warping diagnostic on DC itself.
        "dc_gate_deg": _angle_deg(
            dc_before,
            dc_after,
        ),
        # Cumulative deviation from raw masked Mean after iFFT + time mean.
        "mean_to_time_mean_deg": _angle_deg(
            mean_input,
            time_mean,
        ),
        # Local change introduced by post-pool normalization.
        "time_mean_to_postnorm_deg": _angle_deg(
            pooled_pre,
            pooled_postnorm,
        ),
        # Local change introduced by the trainable output projection.
        "postnorm_to_projection_deg": _angle_deg(
            pooled_postnorm,
            pooled_final,
        ),
        # Final cumulative FLaG angle relative to raw Mean.
        "mean_to_final_deg": _angle_deg(
            mean_input,
            pooled_final,
        ),
    }


@torch.no_grad()
def collect_stage_angles(model, loader, device):
    if not isinstance(
        model.pool,
        FFTLatentAttentionGatePooling,
    ):
        raise ValueError(
            "stage_angle_audit is only defined for FLaG variants"
        )

    model.eval()
    sums = {}
    sums_sq = {}
    count = 0

    for tok1, tok2, _labels in loader:
        for tokens in (tok1, tok2):
            tokens = {
                k: v.to(device)
                for k, v in tokens.items()
            }

            hidden = model.encoder(
                **tokens
            ).last_hidden_state

            # Populate stage tensors on the pooling module.
            _ = model.pool(
                hidden,
                attention_mask=tokens["attention_mask"],
            )

            batch_metrics = _stage_angle_batch(
                model.pool,
                hidden,
                tokens["attention_mask"],
            )

            batch_n = hidden.size(0)
            count += batch_n

            for key, values in batch_metrics.items():
                values = values.detach().float()
                sums[key] = (
                    sums.get(key, 0.0)
                    + float(values.sum().cpu())
                )
                sums_sq[key] = (
                    sums_sq.get(key, 0.0)
                    + float((values * values).sum().cpu())
                )

    out = {
        "n_sentences": int(count),
    }

    for key in sorted(sums):
        mean = sums[key] / max(count, 1)
        if count > 1:
            var = (
                sums_sq[key]
                - (sums[key] * sums[key]) / count
            ) / (count - 1)
            var = max(var, 0.0)
            std = var ** 0.5
        else:
            std = 0.0

        out[key] = {
            "mean": float(mean),
            "std": float(std),
        }

    return out


def print_stage_angle_snapshot(task, pooling, label, snapshot):
    def m(key):
        return snapshot[key]["mean"]

    print(
        f"[ANGLE][{task.upper()}][{pooling}][{label}] "
        f"n={snapshot['n_sentences']} "
        f"mean->DC={m('mean_to_dc_deg'):.3f}deg "
        f"spec_gate={m('spectral_gate_deg'):.3f}deg "
        f"DC_gate={m('dc_gate_deg'):.3f}deg "
        f"mean->timeMean={m('mean_to_time_mean_deg'):.3f}deg "
        f"timeMean->postNorm={m('time_mean_to_postnorm_deg'):.3f}deg "
        f"postNorm->proj={m('postnorm_to_projection_deg'):.3f}deg "
        f"mean->final={m('mean_to_final_deg'):.3f}deg",
        flush=True,
    )


def save_stage_angle_audit(run_dir, audit):
    with (run_dir / "stage_angles.json").open("w") as f:
        json.dump(audit, f, indent=2)


# =========================================================
# Optimizer
# =========================================================

def build_e2e_adam(model, backbone, backbone_lr, new_lr):
    backbone_ids = {id(p) for p in backbone.parameters()}

    backbone_params = [
        p
        for p in model.parameters()
        if id(p) in backbone_ids and p.requires_grad
    ]
    new_params = [
        p
        for p in model.parameters()
        if id(p) not in backbone_ids and p.requires_grad
    ]

    return torch.optim.Adam(
        [
            {
                "params": backbone_params,
                "lr": backbone_lr,
            },
            {
                "params": new_params,
                "lr": new_lr,
            },
        ],
        weight_decay=0.0,
    )


# =========================================================
# Task runners
# =========================================================

def run_imdb(args, device, tokenizer, run_dir):
    raw = load_from_disk(args.imdb_data_path)
    train_split, val_split = split_90_10(
        raw["train"],
        args.seed,
    )
    test_split = raw["test"]

    collator = IMDBCollator(tokenizer, 512)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        IMDBDataset(train_split),
        batch_size=8,
        shuffle=True,
        generator=g,
        collate_fn=collator,
        num_workers=0,
    )
    val_loader = DataLoader(
        IMDBDataset(val_split),
        batch_size=16,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    test_loader = DataLoader(
        IMDBDataset(test_split),
        batch_size=16,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model = IMDBModel(
        args.model_path,
        args.pooling,
        args.mean_anchor_beta_init,
        flag_time_pool=args.flag_time_pool,
        flag_post_pool_norm=(
            not args.disable_post_pool_norm
        ),
    ).to(device)

    optimizer = build_e2e_adam(
        model,
        model.encoder,
        1e-5,
        1e-3,
    )
    criterion = nn.CrossEntropyLoss()

    best_score = -float("inf")
    best_epoch = None
    best_path = run_dir / "best_model.pt"
    history = []

    for epoch in range(1, 4):
        model.train()
        total_loss = 0.0
        total = 0

        for tokens, labels in train_loader:
            tokens = {k: v.to(device) for k, v in tokens.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total += bs

        val = eval_imdb(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "val_accuracy": val["accuracy"],
            "val_f1": val["f1"],
        }
        history.append(row)

        print(
            f"[IMDB][{args.pooling}][s{args.seed}] "
            f"epoch={epoch} loss={row['train_loss']:.6f} "
            f"val_acc={val['accuracy']:.6f} "
            f"val_f1={val['f1']:.6f}",
            flush=True,
        )

        if val["accuracy"] > best_score:
            best_score = val["accuracy"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(
        torch.load(best_path, map_location=device)
    )
    val = eval_imdb(model, val_loader, device)
    test = eval_imdb(model, test_loader, device)

    result = {
        "task": "imdb",
        "protocol": "Kewei2023/Pooling-img-text IMDB E2E",
        "pooling": args.pooling,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "val_accuracy": val["accuracy"],
        "val_f1": val["f1"],
        "test_accuracy": test["accuracy"],
        "test_f1": test["f1"],
        "epochs": 3,
        "max_length": 512,
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "optimizer": "Adam",
        "backbone_lr": 1e-5,
        "new_params_lr": 1e-3,
        "weight_decay": 0.0,
        "scheduler": None,
        "pool_dropout": 0.0,
        "post_pool_norm": (
            not args.disable_post_pool_norm
        ),
        "flag_time_pool": args.flag_time_pool,
        "gate_parameterization": (
            "centered_sigmoid"
            if args.pooling in {
                "FLaG_B1",
                "FLaG_A1Z",
                "FLaG_B2",
            }
            else "residual_sigmoid"
        ),
        "identity_time_out_proj": (
            args.pooling in {
                "FLaG_A1",
                "FLaG_B1",
                "FLaG_A1Z",
                "FLaG_B2",
            }
        ),
        "zero_init_gate_output": (
            args.pooling in {"FLaG_A1Z", "FLaG_B2"}
        ),
        "time_out_proj_init_scale": (
            1.0
            if args.pooling in {
                "FLaG_A1",
                "FLaG_B1",
                "FLaG_A1Z",
                "FLaG_B2",
            }
            else None
        ),
        "split_seed": args.seed,
    }

    return model, result, history


def run_stsb(args, device, tokenizer, run_dir):
    raw = load_from_disk(args.stsb_data_path)

    collator = PairCollator(
        tokenizer,
        max_length=128,
        label_dtype=torch.float32,
    )

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PairDataset(raw["train"], "stsb"),
        batch_size=32,
        shuffle=True,
        generator=g,
        collate_fn=collator,
        num_workers=0,
    )
    val_loader = DataLoader(
        PairDataset(raw["validation"], "stsb"),
        batch_size=64,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    test_loader = DataLoader(
        PairDataset(raw["test"], "stsb"),
        batch_size=64,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model = SentencePairModel(
        args.model_path,
        args.pooling,
        freeze_backbone=False,
        mean_anchor_beta_init=args.mean_anchor_beta_init,
        flag_time_pool=args.flag_time_pool,
        flag_post_pool_norm=(
            not args.disable_post_pool_norm
        ),
    ).to(device)

    optimizer = build_e2e_adam(
        model,
        model.encoder,
        1e-5,
        1e-3,
    )

    best_score = -float("inf")
    best_epoch = None
    best_path = run_dir / "best_model.pt"
    history = []
    angle_audit = {}

    if args.stage_angle_audit:
        angle_audit["epoch0"] = collect_stage_angles(
            model,
            val_loader,
            device,
        )
        print_stage_angle_snapshot(
            "stsb",
            args.pooling,
            "epoch0",
            angle_audit["epoch0"],
        )

    for epoch in range(1, 4):
        model.train()
        total_loss = 0.0
        total = 0

        for tok1, tok2, labels in train_loader:
            tok1 = {k: v.to(device) for k, v in tok1.items()}
            tok2 = {k: v.to(device) for k, v in tok2.items()}
            target = labels.to(device) / 5.0

            optimizer.zero_grad()
            pred, _ = model(tok1, tok2)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total += bs

        val = eval_stsb(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "val_spearman": val["spearman"],
            "val_pearson": val["pearson"],
            "scale": float(
                torch.exp(model.log_scale).detach().cpu()
            ),
            "bias": float(model.bias.detach().cpu()),
        }
        history.append(row)

        print(
            f"[STSB][{args.pooling}][s{args.seed}] "
            f"epoch={epoch} loss={row['train_loss']:.6f} "
            f"val_sp={val['spearman']:.6f} "
            f"val_pe={val['pearson']:.6f}",
            flush=True,
        )

        if args.stage_angle_audit and epoch == 1:
            angle_audit["epoch1"] = collect_stage_angles(
                model,
                val_loader,
                device,
            )
            print_stage_angle_snapshot(
                "stsb",
                args.pooling,
                "epoch1",
                angle_audit["epoch1"],
            )

        if val["spearman"] > best_score:
            best_score = val["spearman"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(
        torch.load(best_path, map_location=device)
    )
    val = eval_stsb(model, val_loader, device)
    test = eval_stsb(model, test_loader, device)

    if args.stage_angle_audit:
        angle_audit["best"] = collect_stage_angles(
            model,
            val_loader,
            device,
        )
        angle_audit["best_epoch"] = int(best_epoch)
        print_stage_angle_snapshot(
            "stsb",
            args.pooling,
            f"best_e{best_epoch}",
            angle_audit["best"],
        )
        save_stage_angle_audit(
            run_dir,
            angle_audit,
        )

    result = {
        "task": "stsb",
        "protocol": "Kewei2023/Pooling-img-text STS E2E",
        "pooling": args.pooling,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "val_spearman": val["spearman"],
        "val_pearson": val["pearson"],
        "test_spearman": test["spearman"],
        "test_pearson": test["pearson"],
        "epochs": 3,
        "max_length": 128,
        "train_batch_size": 32,
        "eval_batch_size": 64,
        "optimizer": "Adam",
        "backbone_lr": 1e-5,
        "new_params_lr": 1e-3,
        "weight_decay": 0.0,
        "scheduler": None,
        "score_mode": "cosine_affine",
        "pool_dropout": 0.0,
        "post_pool_norm": (
            not args.disable_post_pool_norm
        ),
        "flag_time_pool": args.flag_time_pool,
        "gate_parameterization": (
            "centered_sigmoid"
            if args.pooling in {
                "FLaG_B1",
                "FLaG_A1Z",
                "FLaG_B2",
            }
            else "residual_sigmoid"
        ),
        "identity_time_out_proj": (
            args.pooling in {
                "FLaG_A1",
                "FLaG_B1",
                "FLaG_A1Z",
                "FLaG_B2",
            }
        ),
        "zero_init_gate_output": (
            args.pooling in {"FLaG_A1Z", "FLaG_B2"}
        ),
        "time_out_proj_init_scale": (
            1.0
            if args.pooling in {
                "FLaG_A1",
                "FLaG_B1",
                "FLaG_A1Z",
                "FLaG_B2",
            }
            else None
        ),
        "scale": float(
            torch.exp(model.log_scale).detach().cpu()
        ),
        "bias": float(model.bias.detach().cpu()),
        "stage_angle_audit": bool(
            args.stage_angle_audit
        ),
    }

    return model, result, history


def run_sprint(args, device, tokenizer, run_dir):
    raw = load_from_disk(args.sprint_data_path)

    # Our saved adaptation dataset stores a prior 90/10 split of the same
    # official HF validation set. Recombine it, then re-split with the model
    # seed so the protocol matches Pooling-img-text (split_seed=training.seed).
    full_validation = concatenate_datasets(
        [raw["train"], raw["validation"]]
    )
    train_split, val_split = split_90_10(
        full_validation,
        args.seed,
    )
    test_split = raw["test"]

    collator = PairCollator(
        tokenizer,
        max_length=128,
        label_dtype=torch.float32,
    )

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PairDataset(train_split, "sprint"),
        batch_size=32,
        shuffle=True,
        generator=g,
        collate_fn=collator,
        num_workers=0,
    )
    val_loader = DataLoader(
        PairDataset(val_split, "sprint"),
        batch_size=64,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    test_loader = DataLoader(
        PairDataset(test_split, "sprint"),
        batch_size=64,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    model = SentencePairModel(
        args.model_path,
        args.pooling,
        freeze_backbone=True,
        mean_anchor_beta_init=args.mean_anchor_beta_init,
        flag_time_pool=args.flag_time_pool,
        flag_post_pool_norm=(
            not args.disable_post_pool_norm
        ),
    ).to(device)

    trainable = [
        p for p in model.parameters()
        if p.requires_grad
    ]
    optimizer = torch.optim.Adam(
        trainable,
        lr=1e-3,
        weight_decay=0.0,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_score = -float("inf")
    best_epoch = None
    best_path = run_dir / "best_model.pt"
    history = []
    angle_audit = {}

    if args.stage_angle_audit:
        angle_audit["epoch0"] = collect_stage_angles(
            model,
            val_loader,
            device,
        )
        print_stage_angle_snapshot(
            "sprint",
            args.pooling,
            "epoch0",
            angle_audit["epoch0"],
        )

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        total = 0

        for tok1, tok2, labels in train_loader:
            tok1 = {k: v.to(device) for k, v in tok1.items()}
            tok2 = {k: v.to(device) for k, v in tok2.items()}
            target = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(tok1, tok2)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total += bs

        val = eval_sprint(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "val_accuracy": val["accuracy"],
            "val_average_precision": val["average_precision"],
            "val_f1": val["f1"],
            "scale": float(
                torch.exp(model.log_scale).detach().cpu()
            ),
            "bias": float(model.bias.detach().cpu()),
        }
        history.append(row)

        print(
            f"[Sprint][{args.pooling}][s{args.seed}] "
            f"epoch={epoch} loss={row['train_loss']:.6f} "
            f"val_acc={val['accuracy']:.6f} "
            f"val_ap={val['average_precision']:.6f} "
            f"val_f1={val['f1']:.6f}",
            flush=True,
        )

        if args.stage_angle_audit and epoch == 1:
            angle_audit["epoch1"] = collect_stage_angles(
                model,
                val_loader,
                device,
            )
            print_stage_angle_snapshot(
                "sprint",
                args.pooling,
                "epoch1",
                angle_audit["epoch1"],
            )

        checkpoint_key = (
            "average_precision"
            if args.sprint_selection_metric == "ap"
            else "accuracy"
        )
        if val[checkpoint_key] > best_score:
            best_score = val[checkpoint_key]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(
        torch.load(best_path, map_location=device)
    )
    val = eval_sprint(model, val_loader, device)
    test = eval_sprint(model, test_loader, device)

    if args.stage_angle_audit:
        angle_audit["best"] = collect_stage_angles(
            model,
            val_loader,
            device,
        )
        angle_audit["best_epoch"] = int(best_epoch)
        print_stage_angle_snapshot(
            "sprint",
            args.pooling,
            f"best_e{best_epoch}",
            angle_audit["best"],
        )
        save_stage_angle_audit(
            run_dir,
            angle_audit,
        )

    result = {
        "task": "sprint",
        "protocol": "Kewei2023/Pooling-img-text Sprint frozen pooling-only",
        "pooling": args.pooling,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "val_accuracy": val["accuracy"],
        "val_average_precision": val["average_precision"],
        "val_f1": val["f1"],
        "test_accuracy": test["accuracy"],
        "test_average_precision": test["average_precision"],
        "test_f1": test["f1"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "epochs": 10,
        "checkpoint_selection_metric": (
            "val_average_precision"
            if args.sprint_selection_metric == "ap"
            else "val_accuracy"
        ),
        "max_length": 128,
        "train_batch_size": 32,
        "eval_batch_size": 64,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "scheduler": None,
        "backbone_frozen": True,
        "score_mode": "cosine_logit",
        "decision_threshold": 0.5,
        "pool_dropout": 0.0,
        "post_pool_norm": (
            not args.disable_post_pool_norm
        ),
        "flag_time_pool": args.flag_time_pool,
        "gate_parameterization": (
            "centered_sigmoid"
            if args.pooling in {
                "FLaG_B1",
                "FLaG_A1Z",
                "FLaG_B2",
            }
            else "residual_sigmoid"
        ),
        "identity_time_out_proj": (
            args.pooling in {
                "FLaG_A1",
                "FLaG_B1",
                "FLaG_A1Z",
                "FLaG_B2",
            }
        ),
        "zero_init_gate_output": (
            args.pooling in {"FLaG_A1Z", "FLaG_B2"}
        ),
        "time_out_proj_init_scale": (
            1.0
            if args.pooling in {
                "FLaG_A1",
                "FLaG_B1",
                "FLaG_A1Z",
                "FLaG_B2",
            }
            else None
        ),
        "split_seed": args.seed,
        "sprint_split_source": (
            "recombined local adaptation train+validation "
            "(the full official HF validation set)"
        ),
        "scale": float(
            torch.exp(model.log_scale).detach().cpu()
        ),
        "bias": float(model.bias.detach().cpu()),
        "stage_angle_audit": bool(
            args.stage_angle_audit
        ),
    }

    return model, result, history


def maybe_add_beta(model, result):
    pool = model.pool

    if hasattr(pool, "mean_anchor_beta_unbounded"):
        result["mean_anchor_beta"] = float(
            pool.mean_anchor_beta_unbounded.detach().cpu()
        )
    elif hasattr(pool, "mean_anchor_beta_raw"):
        result["mean_anchor_beta"] = float(
            torch.tanh(
                pool.mean_anchor_beta_raw.detach()
            ).cpu()
        )


def save_outputs(run_dir, result, history):
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(result, f, indent=2)

    if history:
        fieldnames = sorted(
            {
                key
                for row in history
                for key in row.keys()
            }
        )
        with (run_dir / "history.csv").open(
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(history)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=["imdb", "stsb", "sprint"],
        required=True,
    )
    parser.add_argument(
        "--pooling",
        choices=[
            "mean",
            "FLaG",
            "FLaG_A1",
            "FLaG_B1",
            "FLaG_A1Z",
            "FLaG_B2",
            "StaticFLaG_ReIm_B2",
            "StaticFLaG_Diag_B2",
            "MeanProj_B2",
            "MeanProjRand",
            "FLaG_AlignmentAnchor",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--flag_time_pool",
        choices=["max", "mean"],
        default="max",
        help=(
            "Final time-domain pooling used by FLaG/Alignment after iFFT. "
            "Default max reproduces the author text protocol; mean is the "
            "matched mechanism ablation."
        ),
    )
    parser.add_argument(
        "--disable_post_pool_norm",
        action="store_true",
        help=(
            "Disable the post-pooling LayerNorm. This is used by the "
            "A1/A1Z initialization-control experiment so A1Z + mean "
            "can be exactly Mean at initialization."
        ),
    )
    parser.add_argument(
        "--sprint_selection_metric",
        choices=["accuracy", "ap"],
        default="accuracy",
        help=(
            "Sprint checkpoint metric. Legacy default accuracy preserves "
            "old protocol; AP is recommended for the paired mechanism study."
        ),
    )
    parser.add_argument(
        "--stage_angle_audit",
        action="store_true",
        help=(
            "Record validation-set FLaG stage angles at epoch0, "
            "after epoch1, and at the selected best checkpoint."
        ),
    )
    parser.add_argument(
        "--mean_anchor_beta_init",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--model_path",
        default=(
            "/home/data/home/wwr_lumos/"
            "models/roberta-base"
        ),
    )
    parser.add_argument(
        "--imdb_data_path",
        default="data/text/imdb",
    )
    parser.add_argument(
        "--stsb_data_path",
        default="data/text/stsbenchmark",
    )
    parser.add_argument(
        "--sprint_data_path",
        default="data/text/sprintduplicatequestions_adaptation",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/text/author_protocol_5seed",
    )

    args = parser.parse_args()
    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    method_dir = {
        "mean": "mean",
        "FLaG": "flag",
        "FLaG_A1": "flag_a1",
        "FLaG_B1": "flag_b1",
        "FLaG_A1Z": "flag_zero",
        "FLaG_B2": "flag_b2",
        "StaticFLaG_ReIm_B2": "static_reim_b2",
        "StaticFLaG_Diag_B2": "static_diag_b2",
        "MeanProj_B2": "mean_proj_b2",
        "MeanProjRand": "mean_proj_rand",
        "FLaG_AlignmentAnchor": "alignment",
    }[args.pooling]

    run_dir = (
        Path(args.output_dir)
        / args.task
        / method_dir
        / f"seed_{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AUTHOR TEXT PROTOCOL")
    print("task:", args.task)
    print("pooling:", args.pooling)
    print("seed:", args.seed)
    print("device:", device)
    print("output:", run_dir)
    print("FLaG dropout: 0.0")
    print(
        "FLaG post_pool_norm:",
        not args.disable_post_pool_norm,
    )
    print("FLaG time_pool:", args.flag_time_pool)
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
    )

    if args.task == "imdb":
        model, result, history = run_imdb(
            args,
            device,
            tokenizer,
            run_dir,
        )
    elif args.task == "stsb":
        model, result, history = run_stsb(
            args,
            device,
            tokenizer,
            run_dir,
        )
    else:
        model, result, history = run_sprint(
            args,
            device,
            tokenizer,
            run_dir,
        )

    # For the matched static FLaG controls, override generic FLaG flags
    # written by task runners so metrics.json truthfully describes the model.
    static_modes = {
        "StaticFLaG_ReIm_B2": "static_reim",
        "StaticFLaG_Diag_B2": "static_diag",
        "MeanProj_B2": "mean_project",
        "MeanProjRand": "mean_project_random",
    }
    if args.pooling in static_modes:
        result.update({
            "mechanism_family": "FFT-free trainable static FLaG control",
            "static_gate_mode": static_modes[args.pooling],
            "uses_fft": False,
            "uses_latent_attention": False,
            "gate_parameterization": (
                "centered_sigmoid"
                if args.pooling not in {"MeanProj_B2", "MeanProjRand"}
                else "none"
            ),
            "identity_time_out_proj": (args.pooling != "MeanProjRand"),
            "zero_init_gate_output": (
                args.pooling not in {"MeanProj_B2", "MeanProjRand"}
            ),
            "time_out_proj_init_scale": (1.0 if args.pooling != "MeanProjRand" else None),
        })

    maybe_add_beta(model, result)
    save_outputs(run_dir, result, history)

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print(json.dumps(result, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
