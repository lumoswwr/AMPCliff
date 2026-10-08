"""Deep mechanism probe for shared real/imag FLaG channel gates.

No retraining. Uses saved best_model.pt from the mean-pool/no-post-LN
initialization/angle audit. Calibrates a *static* gate from the TRAIN split,
then evaluates causal post-hoc interventions on validation or test examples.

Mathematical identity (for real x with masked zeros, shared gate at every k):
   irfft(gr * Re(rfft(x)) + i*gi * Im(rfft(x)))
    = ((gr+gi)/2)*x[t] + ((gr-gi)/2)*x[(-t) mod N].

This is exact for odd/even N up to floating point, and is numerically
verified against the true FLaG forward on the first batch.
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
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_author_text_protocol import (
    PairCollator,
    PairDataset,
    SentencePairModel,
    seed_everything,
    split_90_10,
)
from probe_flag_gate_counterfactual import scoring, masked_mean, angle, FOLDERS


MODES = (
    "full",
    "train_mean_gate",
    "within_batch_swap",
    "scalar_gate",
    "gate_identity",
    "imag_equals_real",
    "no_pad_fixed_gate",
    "long_pad_fixed_gate",
)


class Streaming:
    """Stores only per-sentence diagnostics, never full hidden representations."""
    def __init__(self):
        self.values = defaultdict(list)
        self.logits_sum = None
        self.logits_sq = None
        self.gate_sum = None
        self.gate_sq = None
        self.vector_count = 0

    def add_channel_variance(self, logits, gate):
        # Streaming [2D] statistics avoid materializing all example gates.
        a = logits.detach().double()
        g = gate.detach().double()
        sums_a = a.sum(dim=0).cpu()
        sums_sq_a = a.square().sum(dim=0).cpu()
        sums_g = g.sum(dim=0).cpu()
        sums_sq_g = g.square().sum(dim=0).cpu()
        if self.logits_sum is None:
            self.logits_sum = sums_a
            self.logits_sq = sums_sq_a
            self.gate_sum = sums_g
            self.gate_sq = sums_sq_g
        else:
            self.logits_sum += sums_a
            self.logits_sq += sums_sq_a
            self.gate_sum += sums_g
            self.gate_sq += sums_sq_g
        self.vector_count += int(a.size(0))

    def add(self, key, x):
        self.values[key].extend(
            x.detach().float().reshape(-1).cpu().tolist()
        )

    def summary(self):
        out = {}
        for key, xs in sorted(self.values.items()):
            a = np.asarray(xs, dtype=np.float64)
            out[key] = {
                "mean": float(a.mean()),
                "median": float(np.median(a)),
                "p10": float(np.quantile(a, 0.1)),
                "p90": float(np.quantile(a, 0.9)),
                "n": int(len(a)),
            }
        if self.vector_count > 1:
            n = self.vector_count
            logits_var = (
                self.logits_sq / n - (self.logits_sum / n).square()
            ).clamp_min(0)
            gate_var = (
                self.gate_sq / n - (self.gate_sum / n).square()
            ).clamp_min(0)
            out["logits_between_sentence_channel_sd_mean"] = {
                "mean": float(logits_var.sqrt().mean()),
                "max": float(logits_var.sqrt().max()),
                "n": n,
            }
            out["gate_between_sentence_channel_sd_mean"] = {
                "mean": float(gate_var.sqrt().mean()),
                "max": float(gate_var.sqrt().max()),
                "n": n,
            }
        return out


def load_checkpoint(args, device):
    folder = FOLDERS[args.pooling]
    path_suffix = Path(args.task) / folder / ("seed_" + str(args.seed))
    run_dir = Path(args.checkpoint_root) / path_suffix
    ckpt = run_dir / "best_model.pt"
    if not ckpt.is_file():
        fallback = Path("outputs/text/flag_init_ablation_seed0") / path_suffix
        if not (fallback / "best_model.pt").is_file():
            raise FileNotFoundError(
                f"Missing checkpoint: {ckpt}; fallback: {fallback / 'best_model.pt'}"
            )
        run_dir = fallback
        ckpt = fallback / "best_model.pt"

    model = SentencePairModel(
        args.model_path, args.pooling,
        freeze_backbone=(args.task == "sprint"),
        mean_anchor_beta_init=0.1,
        flag_time_pool="mean",
        flag_post_pool_norm=False,
    ).to(device)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, run_dir, ckpt


def split_for(args, training):
    data = load_from_disk(
        args.stsb_data_path if args.task == "stsb" else args.sprint_data_path
    )
    if args.task == "stsb":
        return data["train"] if training else data[args.split]
    if args.split == "test" and not training:
        return data["test"]
    full = concatenate_datasets([data["train"], data["validation"]])
    train, valid = split_90_10(full, args.seed)
    return train if training else valid


def dataloader(dataset, task, tokenizer, args, n_samples, offset):
    if n_samples and n_samples < len(dataset):
        rng = np.random.default_rng(args.seed + offset)
        chosen = np.sort(
            rng.choice(len(dataset), size=n_samples, replace=False)
        )
        dataset = dataset.select(chosen.tolist())
    collator = PairCollator(
        tokenizer, max_length=128, label_dtype=torch.float32
    )
    return DataLoader(
        PairDataset(dataset, task),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )


@torch.no_grad()
def get_gate(pool, hidden, mask, return_logits=False):
    spec = pool._to_frequency_tokens(hidden, mask)
    lat = pool._latent_pool_in_frequency(spec)
    hidden_gate = lat.mean(dim=1)
    logits = pool.freq_gate(hidden_gate)
    raw = torch.sigmoid(logits)
    if pool.gate_parameterization == "centered_sigmoid":
        gate = 2.0 * raw
    else:
        gate = 1.0 + raw if pool.gate_residual else raw
    return (gate, logits) if return_logits else gate


def sequence_stats(hidden, mask):
    x = hidden * mask.unsqueeze(-1).to(hidden.dtype)
    N = x.shape[1]
    m = masked_mean(x, mask)
    rev_idx = (-torch.arange(N, device=x.device)) % N
    mr = masked_mean(x.index_select(1, rev_idx), mask)
    L = mask.sum(dim=1, keepdim=True).clamp_min(1)
    h0_over_L = x[:, 0, :] / L
    return m, mr, h0_over_L


def time_mix(m, mr, gate):
    d = m.shape[-1]
    gr, gi = gate[:, :d], gate[:, d:]
    a = .5 * (gr + gi)
    b = .5 * (gr - gi)
    return a * m + b * mr


@torch.no_grad()
def calibration(model, loader, device):
    total_gate = None
    n = 0
    for i, (a, b, _y) in enumerate(loader):
        for tokens in (a, b):
            tokens = {k: v.to(device) for k, v in tokens.items()}
            hidden = model.encoder(**tokens).last_hidden_state
            gate = get_gate(model.pool, hidden, tokens["attention_mask"])
            gs = gate.double().sum(dim=0, keepdim=True)
            total_gate = gs if total_gate is None else total_gate + gs
            n += gate.size(0)
        if i % 50 == 0:
            print(f"[calibration] batches={i+1}/{len(loader)}", flush=True)
    if n == 0:
        raise RuntimeError("Empty training calibration dataset")
    gate = (total_gate / n).to(dtype=torch.float32, device=device)
    return gate, n


@torch.no_grad()
def get_outputs(pool, hidden, mask, train_gate, rec, check):
    gate, logits = get_gate(pool, hidden, mask, return_logits=True)
    rec.add_channel_variance(logits, gate)
    if gate.shape[-1] != 2 * hidden.shape[-1]:
        raise RuntimeError("Unexpected gate width (must be 2*D)")

    m, mr, h0_over_L = sequence_stats(hidden, mask)
    d = m.shape[-1]
    gr, gi = gate[:, :d], gate[:, d:]
    mean_gate = train_gate.expand_as(gate)

    # Counterfactuals in closed-form time space; same learned projection.
    altered = {
        "full": time_mix(m, mr, gate),
        "train_mean_gate": time_mix(m, mr, mean_gate),
        "within_batch_swap": time_mix(
            m, mr,
            torch.roll(gate, shifts=max(1, gate.shape[0] // 2), dims=0)
            if gate.size(0) > 1 else mean_gate,
        ),
        "scalar_gate": time_mix(
            m, mr, gate.mean(dim=-1, keepdim=True).expand_as(gate)
        ),
        "gate_identity": m,
        "imag_equals_real": gr * m,
        "no_pad_fixed_gate": gr * m,
        # Under at least 2L right-padding, the masked reversal term is h0/L.
        "long_pad_fixed_gate": .5 * (gr + gi) * m
                              + .5 * (gr - gi) * h0_over_L,
    }
    out = {k: pool.time_out_proj(z) for k, z in altered.items()}

    if check:
        actual = pool(hidden, attention_mask=mask)
        err = (actual - out["full"]).abs().max().item()
        print(f"[identity] direct-time vs FFT FLaG max_abs_error={err:.3e}", flush=True)
        if not torch.allclose(actual, out["full"], atol=2e-4, rtol=2e-4):
            raise RuntimeError(
                "Closed-form real/imag reversal is not equal to actual FLaG!"
            )
        if not torch.allclose(
            out["imag_equals_real"], out["no_pad_fixed_gate"],
            atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError("No-pad and real=imag control mismatch.")

    # Log energy geometry: DC is the sum of masked token embeddings.
    dc = m
    dc_energy = dc.square()
    total_e = dc_energy.sum(dim=-1).clamp_min(1e-12)
    sig_real = (
        (gr - 1) if pool.gate_parameterization == "residual_sigmoid"
        else gr / 2
    )
    sig_imag = (
        (gi - 1) if pool.gate_parameterization == "residual_sigmoid"
        else gi / 2
    )
    suppressed = sig_real < 0.05
    suppressed_imag = sig_imag < 0.05
    energy_suppressed = (
        dc_energy * suppressed
    ).sum(dim=-1) / total_e
    k = max(1, math.ceil(.05 * d))
    lowest = gr.argsort(dim=-1)[:, :k]
    low_energy = dc_energy.gather(1, lowest).sum(dim=-1) / total_e

    # Diagonal energy identity, including DC-gate theoretical cosine.
    weights = dc_energy / total_e.unsqueeze(-1)
    gbar = (weights * gr).sum(-1)
    g2bar = (weights * gr.square()).sum(-1)
    cosine = (gbar / g2bar.sqrt().clamp_min(1e-12)).clamp(-1., 1.)
    predicted_angle = torch.rad2deg(torch.acos(cosine))

    rec.add("dc_angle_theoretical_deg", predicted_angle)
    rec.add("suppressed_real_channel_fraction", suppressed.float().mean(-1))
    rec.add("suppressed_imag_channel_fraction", suppressed_imag.float().mean(-1))
    rec.add("suppressed_real_dc_energy_fraction", energy_suppressed)
    rec.add("lowest_5pct_real_gate_dc_energy_fraction", low_energy)
    rec.add("dc_weighted_gate_mean", gbar)
    rec.add("dc_weighted_gate_sd", (g2bar - gbar.square()).clamp_min(0).sqrt())
    rec.add("real_imag_gate_absolute_difference", (gr - gi).abs().mean(-1))
    rec.add("real_imag_gate_channel_cosine", F.cosine_similarity(gr, gi, dim=-1))
    # Machine-precision diagnostics. Earlier result printing rounded values
    # to five decimals, which is insufficient to claim exact gate invariance.
    delta_gate = (gate - mean_gate).abs()
    rec.add("dynamic_gate_max_abs_delta_vs_train_mean", delta_gate.amax(-1))
    rec.add("dynamic_gate_bitwise_diff_fraction", (gate != mean_gate).float().mean(-1))
    rec.add("dynamic_gate_fraction_above_1e-7", (delta_gate > 1e-7).float().mean(-1))
    rec.add("dynamic_gate_fraction_above_1e-5", (delta_gate > 1e-5).float().mean(-1))
    rec.add("gate_logits_within_sentence_channel_sd", logits.std(-1, unbiased=False))
    rec.add("gate_logits_within_sentence_abs_mean", logits.abs().mean(-1))
    rec.add("dynamic_gate_abs_delta_vs_train_mean", (gate - mean_gate).abs().mean(-1))
    rec.add("dynamic_gate_l2_delta_vs_train_mean", (
        gate - mean_gate
    ).norm(dim=-1) / gate.norm(dim=-1).clamp_min(1e-8))
    # Binary channel-selection stability against TRAIN-derived fixed gate.
    threshold = 1.5 if pool.gate_parameterization == "residual_sigmoid" else 1.
    rec.add("dynamic_gate_mask_flip_fraction_vs_train_mean", (
        (gate >= threshold) != (mean_gate >= threshold)
    ).float().mean(-1))
    rec.add("valid_length_fraction", mask.float().mean(-1))
    rec.add("native_to_no_pad_angle_deg", angle(out["full"], out["no_pad_fixed_gate"]))
    rec.add("native_to_long_pad_angle_deg", angle(out["full"], out["long_pad_fixed_gate"]))
    # A reversal term strongly linked to position 0, often RoBERTa's <s>.
    rec.add("reversal_vs_original_masked_mean_angle_deg", angle(m, mr))
    rec.add("reversal_mean_over_original_norm", mr.norm(dim=-1) / m.norm(dim=-1).clamp_min(1e-8))
    return out


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["stsb", "sprint"], required=True)
    parser.add_argument("--pooling", choices=tuple(FOLDERS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--train_pairs", type=int, default=512,
                        help="Number of TRAIN pairs to estimate static gate; 0=all.")
    parser.add_argument("--eval_pairs", type=int, default=0,
                        help="Number of EVAL pairs; 0=all.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_path", default="/home/data/home/wwr_lumos/models/roberta-base")
    parser.add_argument("--stsb_data_path", default="data/text/stsbenchmark")
    parser.add_argument("--sprint_data_path", default="data/text/sprintduplicatequestions_adaptation")
    parser.add_argument("--checkpoint_root", default="outputs/text/flag_stage_angle_audit_seed0")
    parser.add_argument("--output_root", default="outputs/text/flag_deep_mechanism_seed0")
    args = parser.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, src, checkpoint = load_checkpoint(args, device)
    if not model.pool.use_gate or not model.pool.use_latent:
        raise ValueError("Expected latent FLaG gate.")
    if model.pool.post_pool_norm or model.pool.time_pool != "mean":
        raise ValueError("Probe is only valid for mean-time pooling without post LN.")
    if model.pool.dropout.p != 0:
        raise ValueError("Probe requires dropout=0.")
    if (
        model.pool.window_type is not None
        or model.pool.fixed_fft_length is not None
        or model.pool.remove_dc
        or model.pool.dc_only
        or model.pool.learned_frequency_gate
        or model.pool.attention_frequency_gate
    ):
        raise ValueError("Probe assumes unwindowed, unaltered standard FLaG.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    train_loader = dataloader(
        split_for(args, True), args.task, tokenizer, args,
        args.train_pairs, offset=7341,
    )
    test_loader = dataloader(
        split_for(args, False), args.task, tokenizer, args,
        args.eval_pairs, offset=9017,
    )
    print(f"[deep] task={args.task} method={args.pooling} device={device}", flush=True)
    print(f"[deep] source checkpoint={checkpoint}", flush=True)
    print(f"[deep] calibration_pairs={len(train_loader.dataset)} evaluation_pairs={len(test_loader.dataset)}", flush=True)
    train_gate, ncal = calibration(model, train_loader, device)
    print(f"[deep] train_mean_gate computed from {ncal} sentences", flush=True)

    collector = Streaming()
    predictions = {mode: [] for mode in MODES}
    labels = []
    scale, bias = model.log_scale.exp(), model.bias
    checked = False

    for i, (ta, tb, yy) in enumerate(test_loader):
        zz = []
        for tok in (ta, tb):
            tok = {k: v.to(device) for k, v in tok.items()}
            hidden = model.encoder(**tok).last_hidden_state
            results = get_outputs(
                model.pool, hidden, tok["attention_mask"],
                train_gate, collector, check=not checked,
            )
            checked = True
            zz.append({mode: F.normalize(z, dim=-1, p=2) for mode, z in results.items()})
        for mode in MODES:
            sc = scale * (zz[0][mode] * zz[1][mode]).sum(-1) + bias
            predictions[mode].extend(sc.cpu().tolist())
        labels.extend(yy.cpu().tolist())
        if i % 50 == 0:
            print(f"[deep] evaluated {i+1}/{len(test_loader)} batches", flush=True)

    truth = np.asarray(labels)
    full_scores = np.asarray(predictions["full"])
    per_mode = {}
    for mode in MODES:
        sc = np.asarray(predictions[mode])
        perf = scoring(args.task, truth, sc)
        perf["score_rank_spearman_vs_full"] = float(spearmanr(full_scores, sc).statistic)
        perf["mean_abs_score_delta_vs_full"] = float(np.abs(full_scores-sc).mean())
        per_mode[mode] = perf

    metpath = src / "metrics.json"
    metrics = json.loads(metpath.read_text()) if metpath.is_file() else {}
    if args.split == "validation" and args.eval_pairs == 0 and metrics:
        primary = "spearman" if args.task == "stsb" else "average_precision"
        expected = metrics.get("val_" + primary)
        if expected is not None:
            delta = abs(expected - per_mode["full"][primary])
            print(f"[deep] validation baseline guard original={expected:.6f}, measured={per_mode['full'][primary]:.6f}, error={delta:.3g}", flush=True)
            if delta > .002:
                raise RuntimeError("Full mode not matching previous validation metric.")

    output = {
        "task": args.task, "pooling": args.pooling,
        "split": args.split, "seed": args.seed,
        "checkpoint": str(checkpoint),
        "n_calibration_train_sentences": ncal,
        "n_evaluation_pairs": len(truth),
        "calibrated_static_gate_summary": {
            "min": float(train_gate.min()),
            "max": float(train_gate.max()),
            "std_channels": float(train_gate.std(unbiased=False)),
            "real_imag_mean_abs_diff": float(
                (train_gate[:, :train_gate.shape[-1]//2] -
                 train_gate[:, train_gate.shape[-1]//2:]).abs().mean()
            ),
        },
        "mode_performance": per_mode,
        "diagnostics": collector.summary(),
    }
    folder = FOLDERS[args.pooling]
    destdir = Path(args.output_root) / args.task / folder / f"seed_{args.seed}"
    destdir.mkdir(parents=True, exist_ok=True)
    dest = destdir / f"deep_{args.split}.json"
    dest.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")

    primary = "spearman" if args.task == "stsb" else "average_precision"
    print(f"\n{'Mode':24s} {primary:>10s} {'Δ primary':>12s} {'rank vs full':>13s}")
    base = per_mode["full"][primary]
    for mode in MODES:
        p = per_mode[mode]
        print(f"{mode:24s} {p[primary]:10.6f} {p[primary]-base:+12.6f} {p['score_rank_spearman_vs_full']:13.4f}")
    print("\nKey channel-energy / padding diagnostics:")
    diag = output["diagnostics"]
    for key in (
        "dc_angle_theoretical_deg",
        "suppressed_real_channel_fraction",
        "suppressed_imag_channel_fraction",
        "suppressed_real_dc_energy_fraction",
        "lowest_5pct_real_gate_dc_energy_fraction",
        "real_imag_gate_absolute_difference",
        "dynamic_gate_abs_delta_vs_train_mean",
        "dynamic_gate_max_abs_delta_vs_train_mean",
        "dynamic_gate_bitwise_diff_fraction",
        "dynamic_gate_fraction_above_1e-7",
        "dynamic_gate_fraction_above_1e-5",
        "logits_between_sentence_channel_sd_mean",
        "gate_between_sentence_channel_sd_mean",
        "dynamic_gate_mask_flip_fraction_vs_train_mean",
        "native_to_no_pad_angle_deg",
        "native_to_long_pad_angle_deg",
    ):
        print(f"  {key:47s} {diag[key]['mean']:.6f}")
    print(f"[deep] saved={dest}", flush=True)


if __name__ == "__main__":
    main()
