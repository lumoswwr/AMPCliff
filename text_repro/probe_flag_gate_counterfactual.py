"""Checkpoint-only FLaG gate intervention, matched to prior mean-pool/no-LN run.

Full vs: DC-off, AC-off, gate-off, scalar gate, bias-only gate,
feature-only gate, no output projection, raw Mean (same encoder).
No retraining, and the full forward/validation score is verified.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_from_disk
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_author_text_protocol import (
    PairCollator, PairDataset, SentencePairModel,
    seed_everything, split_90_10,
)

MODES = (
    "full", "dc_identity", "ac_identity", "gate_identity", "scalar_gate",
    "bias_only", "feature_only", "no_projection", "raw_mean",
)
FOLDERS = {
    "FLaG": "flag",
    "FLaG_A1": "flag_a1",
    "FLaG_B1": "flag_b1",
    "FLaG_B2": "flag_b2",
}


def masked_mean(x, mask):
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp_min(1e-6)


def angle(a, b):
    cos = F.cosine_similarity(a, b, dim=-1, eps=1e-8)
    return torch.rad2deg(torch.acos(cos.clamp(-1., 1.)))


class Collector:
    def __init__(self):
        self.sums = defaultdict(float)
        self.sqs = defaultdict(float)
        self.counts = defaultdict(int)
        self.logit_sum = None
        self.logit_sq = None
        self.nlogits = 0

    def add(self, name, x):
        v = x.detach().double().reshape(-1)
        self.sums[name] += float(v.sum())
        self.sqs[name] += float(v.square().sum())
        self.counts[name] += v.numel()

    def add_logits(self, x):
        v = x.detach().double()
        s, q = v.sum(0).cpu(), v.square().sum(0).cpu()
        if self.logit_sum is None:
            self.logit_sum, self.logit_sq = s, q
        else:
            self.logit_sum += s
            self.logit_sq += q
        self.nlogits += v.shape[0]

    def output(self):
        out = {}
        for key, n in self.counts.items():
            mean = self.sums[key] / n
            sd = math.sqrt(max(0., self.sqs[key] / n - mean * mean))
            out[key] = {"mean": mean, "std": sd, "n": n}
        if self.nlogits > 1:
            m = self.logit_sum / self.nlogits
            var = (self.logit_sq / self.nlogits - m.square()).clamp_min(0)
            out["logit_between_sentence_sd"] = {
                "mean": float(var.sqrt().mean()), "n": self.nlogits
            }
        return out


@torch.no_grad()
def interventions(pool, hidden, mask, collector):
    if pool.time_pool != "mean" or pool.post_pool_norm or pool.dropout.p != 0:
        raise ValueError("Probe expects time_pool=mean, no post-pool LN and dropout=0.")
    if (pool.remove_dc or pool.dc_only or pool.learned_frequency_gate
            or pool.attention_frequency_gate or not pool.use_latent):
        raise ValueError("Unexpected FLaG architecture for this probe.")

    spec = pool._to_frequency_tokens(hidden, mask)
    latent = pool._latent_pool_in_frequency(spec)
    s = latent.mean(dim=1)
    h1 = pool.freq_gate[0](s)
    h2 = pool.freq_gate[1](h1)
    logits = pool.freq_gate[3](pool.freq_gate[2](h2))
    b = pool.freq_gate[3].bias
    if b is None:
        raise ValueError("Expected gate output bias.")

    def gate(v):
        raw = torch.sigmoid(v)
        if pool.gate_parameterization == "centered_sigmoid":
            delta = 2 * raw - 1
        else:
            delta = raw
        return 1 + delta if pool.gate_residual else raw

    g = gate(logits)
    d = hidden.shape[-1]
    dc = spec[:, 0, :d]
    new_dc = dc * g[:, :d]

    # For a positive diagonal gate, the DC cosine is given exactly by
    # channel-energy-weighted moments of g (independent of FFT).
    w = dc.square()
    wsum = w.sum(-1).clamp_min(1e-20)
    avg = (w * g[:, :d]).sum(-1) / wsum
    avg2 = (w * g[:, :d].square()).sum(-1) / wsum
    theoretical_cos = avg / avg2.sqrt().clamp_min(1e-20)
    measured_cos = F.cosine_similarity(dc, new_dc, dim=-1)
    if float((theoretical_cos - measured_cos).abs().max()) > 2e-4:
        raise RuntimeError("DC cosine identity check failed.")

    collector.add("dc_angle_deg", angle(dc, new_dc))
    collector.add("gate_std_across_channels", g.std(-1, unbiased=False))
    collector.add("gate_cv_across_channels", g.std(-1, unbiased=False) / g.mean(-1).clamp_min(1e-8))
    collector.add("gate_fraction_below_1", (g < 1).float().mean(-1))
    collector.add("sigmoid_fraction_below_.05", (torch.sigmoid(logits) < .05).float().mean(-1))
    collector.add("sigmoid_fraction_above_.95", (torch.sigmoid(logits) > .95).float().mean(-1))
    collector.add("gate_input_channel_sd", s.std(-1, unbiased=False))
    collector.add("W1_channel_sd", h1.std(-1, unbiased=False))
    collector.add("GELU_channel_sd", h2.std(-1, unbiased=False))
    collector.add("logits_channel_sd", logits.std(-1, unbiased=False))
    collector.add("feature_logits_l2_vs_full", (logits-b).norm(dim=-1) / logits.norm(dim=-1).clamp_min(1e-8))
    collector.add_logits(logits)

    def decode(enhanced, no_projection=False):
        time = pool._back_to_time(enhanced, seq_len=hidden.size(1))
        p = masked_mean(time, mask)
        return p if no_projection else pool.time_out_proj(p)

    full_spec = spec * g.unsqueeze(1)
    dc_off = full_spec.clone()
    dc_off[:, 0, :] = spec[:, 0, :]
    ac_off = full_spec.clone()
    ac_off[:, 1:, :] = spec[:, 1:, :]
    scalar = g.mean(-1, keepdim=True).expand_as(g)
    outputs = {
        "full": decode(full_spec),
        "dc_identity": decode(dc_off),
        "ac_identity": decode(ac_off),
        "gate_identity": decode(spec),
        "scalar_gate": decode(spec * scalar.unsqueeze(1)),
        "bias_only": decode(spec * gate(b.unsqueeze(0)).unsqueeze(1)),
        "feature_only": decode(spec * gate(logits-b).unsqueeze(1)),
        "no_projection": decode(full_spec, no_projection=True),
        "raw_mean": masked_mean(hidden, mask),
    }
    for key, output in outputs.items():
        collector.add("mean_to_" + key + "_deg", angle(outputs["raw_mean"], output))
    return outputs


def loader_for(args, tokenizer):
    data = load_from_disk(args.stsb_data_path if args.task == "stsb"
                          else args.sprint_data_path)
    if args.task == "stsb":
        part = data["validation" if args.split == "validation" else "test"]
    elif args.split == "test":
        part = data["test"]
    else:
        full = concatenate_datasets([data["train"], data["validation"]])
        _, part = split_90_10(full, args.seed)

    if args.max_pairs:
        rng = np.random.default_rng(args.seed + 2026)
        n = min(args.max_pairs, len(part))
        indexes = np.sort(rng.choice(len(part), n, replace=False))
        part = part.select(indexes.tolist())

    collate = PairCollator(tokenizer, 128, torch.float32)
    return DataLoader(PairDataset(part, args.task), batch_size=args.batch_size,
                      shuffle=False, collate_fn=collate, num_workers=0)


def scoring(task, y, p):
    if task == "stsb":
        return {
            "spearman": float(spearmanr(y, p).statistic),
            "pearson": float(pearsonr(y, p).statistic),
        }
    probs = 1 / (1 + np.exp(-np.clip(p, -50, 50)))
    cls = probs >= .5
    return {
        "average_precision": float(average_precision_score(y, probs)),
        "accuracy": float(accuracy_score(y, cls)),
        "f1": float(f1_score(y, cls, zero_division=0)),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["stsb", "sprint"], required=True)
    parser.add_argument("--pooling", choices=tuple(FOLDERS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_path", default="/home/data/home/wwr_lumos/models/roberta-base")
    parser.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    parser.add_argument("--sprint_data_path", default="data/text/sprintduplicatequestions_adaptation")
    parser.add_argument("--checkpoint_root", default="outputs/text/flag_stage_angle_audit_seed0")
    parser.add_argument("--output_root", default="outputs/text/flag_gate_counterfactual_seed0")
    args = parser.parse_args()

    seed_everything(args.seed)
    folder = FOLDERS[args.pooling]
    subpath = Path(args.task) / folder / f"seed_{args.seed}"
    source = Path(args.checkpoint_root) / subpath
    checkpoint = source / "best_model.pt"
    if not checkpoint.is_file():
        fallback = Path("outputs/text/flag_init_ablation_seed0") / subpath
        if (fallback / "best_model.pt").is_file():
            source, checkpoint = fallback, fallback / "best_model.pt"
        else:
            raise FileNotFoundError(f"No checkpoint at {checkpoint} or {fallback}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    loader = loader_for(args, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentencePairModel(
        args.model_path, args.pooling, args.task == "sprint", .1,
        flag_time_pool="mean", flag_post_pool_norm=False,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    recorder = Collector()
    preds = {mode: [] for mode in MODES}
    labels = []
    checked = False
    scale, bias = model.log_scale.exp(), model.bias

    print(f"[gate] task={args.task} method={args.pooling} pairs={len(loader.dataset)}")
    print(f"[gate] loaded {checkpoint}")

    for i, (tok1, tok2, y) in enumerate(loader):
        encoded = []
        for tok in (tok1, tok2):
            tok = {k: v.to(device) for k, v in tok.items()}
            hidden = model.encoder(**tok).last_hidden_state
            vals = interventions(model.pool, hidden, tok["attention_mask"], recorder)
            if not checked:
                direct = model.pool(hidden, tok["attention_mask"])
                difference = float((direct - vals["full"]).abs().max())
                print(f"[gate] full forward max_abs_delta={difference:.3e}")
                if not torch.allclose(direct, vals["full"], rtol=2e-5, atol=2e-5):
                    raise RuntimeError("Probe full mode differs from trained forward.")
                checked = True
            encoded.append({k: F.normalize(v, dim=-1, p=2) for k, v in vals.items()})
        for mode in MODES:
            cosine = (encoded[0][mode] * encoded[1][mode]).sum(-1)
            score = scale * cosine + bias
            preds[mode].extend(score.cpu().tolist())
        labels.extend(y.tolist())
        if i % 50 == 0:
            print(f"[gate] batch {i+1}/{len(loader)}", flush=True)

    y = np.asarray(labels)
    full = np.asarray(preds["full"])
    results = {}
    for mode, values in preds.items():
        scores = np.asarray(values)
        result = scoring(args.task, y, scores)
        result["score_spearman_vs_full"] = float(spearmanr(scores, full).statistic)
        result["mean_abs_score_change"] = float(np.abs(scores-full).mean())
        results[mode] = result

    metadata_file = source / "metrics.json"
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    if args.split == "validation" and args.max_pairs == 0 and metadata:
        k = "spearman" if args.task == "stsb" else "average_precision"
        old = metadata.get("val_" + k)
        if old is not None:
            diff = abs(old - results["full"][k])
            print(f"[gate] validation guard: old={old:.6f} probe={results['full'][k]:.6f} delta={diff:.3e}")
            if diff > .002:
                raise RuntimeError("Full-mode result differs from checkpoint validation metric.")

    output = {
        "task": args.task, "pooling": args.pooling, "seed": args.seed,
        "split": args.split, "n_pairs": len(y),
        "positive_prevalence": float(y.mean()) if args.task == "sprint" else None,
        "checkpoint": str(checkpoint),
        "mode_results": results,
        "gate_diagnostics": recorder.output(),
    }
    target = Path(args.output_root) / subpath
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"probe_{args.split}.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")

    k = "spearman" if args.task == "stsb" else "average_precision"
    print(f"\n{'Mode':18s} {k:>12s} {'Rank vs full':>13s} {'Delta score':>13s}")
    for mode in MODES:
        r = results[mode]
        print(f"{mode:18s} {r[k]:12.6f} {r['score_spearman_vs_full']:13.4f} {r['mean_abs_score_change']:13.4f}")
    for key in (
        "dc_angle_deg", "gate_std_across_channels", "gate_cv_across_channels",
        "gate_fraction_below_1", "logits_channel_sd", "logit_between_sentence_sd",
    ):
        v = output["gate_diagnostics"].get(key)
        if v is not None:
            print(f"[gate] {key} = {v['mean']:.6f}")
    print(f"[gate] saved {path}")


if __name__ == "__main__":
    main()
