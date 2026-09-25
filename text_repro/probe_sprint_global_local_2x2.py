import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import load_from_disk
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer

from text_repro.train_sprint import SprintPairClassifier

from AMPCliff.factory.pooling.flag_pooling import (
    FFTLatentAttentionGatePooling,
    STFTLatentAttentionGatePooling,
)

from AMPCliff.factory.pooling.llm_pooling_dropin import (
    masked_max_pooling,
    masked_mean_pooling,
)


ROOT = Path(
    "/home/data/home/wwr_lumos/AMPCliff"
)

MODEL_PATH = Path(
    "/home/data/home/wwr_lumos/models/roberta-base"
)

DATA_PATH = (
    ROOT
    / "data/text/sprintduplicatequestions_adaptation"
)


# ============================================================
# Basic utilities
# ============================================================

def average_precision(scores, labels):
    return float(
        average_precision_score(
            np.asarray(labels, dtype=np.int64),
            np.asarray(scores, dtype=np.float64),
        )
    )


def pair_prediction(
    embeddings,
    batch_size,
):
    return F.cosine_similarity(
        embeddings[:batch_size],
        embeddings[batch_size:],
        dim=-1,
    )


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if (
        np.std(a) == 0
        or np.std(b) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(a, b)[0, 1]
    )


# ============================================================
# Checkpoints
# ============================================================

def checkpoint_dir(source, seed):
    exp = (
        "sprint_frozen_flag"
        if source == "FLaG"
        else "sprint_frozen_e12"
    )

    path = (
        ROOT
        / "outputs/text/sprintduplicatequestions/experiments"
        / exp
        / f"seed_{seed}"
    )

    if not (path / "best_model.pt").exists():
        raise FileNotFoundError(
            path / "best_model.pt"
        )

    if not (path / "metrics.json").exists():
        raise FileNotFoundError(
            path / "metrics.json"
        )

    return path


def build_source_model(
    source,
    ckpt_dir,
    device,
):
    pooling = (
        "FLaG"
        if source == "FLaG"
        else "STFT_FLaG"
    )

    model = SprintPairClassifier(
        model_path=str(MODEL_PATH),
        pooling=pooling,
        stft_win_length=16,
        stft_hop_length=16,
        stft_window_type="rect",
        stft_center=False,
    )

    ckpt = torch.load(
        ckpt_dir / "best_model.pt",
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# Global/local operator copies with identical learned weights
# ============================================================

def make_operator_pools(
    source_pool,
    d_model,
    device,
):
    # Preserve every non-operator choice from the source-trained pool.
    dropout_p = float(
        source_pool.dropout.p
    )

    common = dict(
        d_model=d_model,
        num_latents=source_pool.num_latents,
        num_heads=source_pool.num_heads,
        dropout=dropout_p,
        time_pool=source_pool.time_pool,
        gate_residual=source_pool.gate_residual,
        eps=source_pool.eps,
        use_gate=source_pool.use_gate,
        use_latent=source_pool.use_latent,
        post_pool_norm=source_pool.post_pool_norm,
        fixed_fft_length=None,
    )

    global_pool = FFTLatentAttentionGatePooling(
        **common,
        window_type=None,
    )

    local_pool = STFTLatentAttentionGatePooling(
        **common,
        win_length=16,
        hop_length=16,
        use_frame_positional_encoding=False,
        max_frame_positions=32,
        local_window_type="rect",
        center_frames=False,
    )

    state = source_pool.state_dict()

    global_pool.load_state_dict(
        state,
        strict=True,
    )

    local_pool.load_state_dict(
        state,
        strict=True,
    )

    global_pool = global_pool.to(device)
    local_pool = local_pool.to(device)

    global_pool.eval()
    local_pool.eval()

    return global_pool, local_pool


# ============================================================
# Gate helpers
# ============================================================

def gate_scale(
    pool,
    latent_out,
):
    raw = torch.sigmoid(
        pool.freq_gate(
            latent_out.mean(dim=1)
        )
    )

    if pool.gate_residual:
        scale = 1.0 + raw
    else:
        scale = raw

    return raw, scale


def build_global_frequency(
    pool,
    hidden,
    mask,
):
    x = pool._apply_input_window(
        hidden,
        mask,
    )

    T = hidden.size(1)

    spec = torch.fft.rfft(
        x,
        n=T,
        dim=1,
    )

    return torch.cat(
        [spec.real, spec.imag],
        dim=-1,
    )


def global_gate(
    pool,
    global_freq,
):
    latent = pool._latent_pool_in_frequency(
        global_freq
    )

    return gate_scale(
        pool,
        latent,
    )


def local_gate(
    pool,
    local_freq,
    freq_mask,
    meta,
):
    attention_tokens = (
        pool._add_frame_position_for_attention(
            local_freq,
            freq_mask,
            meta,
        )
    )

    latent = pool._latent_pool_in_frequency(
        attention_tokens,
        freq_attention_mask=freq_mask,
    )

    return gate_scale(
        pool,
        latent,
    )


# ============================================================
# Reconstruction helpers
# ============================================================

def finalize_embedding(
    pool,
    time_tokens,
    mask,
):
    if pool.time_pool == "mean":
        pooled = masked_mean_pooling(
            time_tokens,
            mask,
            eps=pool.eps,
        )

    elif pool.time_pool == "max":
        pooled = masked_max_pooling(
            time_tokens,
            mask,
        )

    else:
        raise RuntimeError(
            pool.time_pool
        )

    if pool.post_pool_norm:
        pooled = pool.norm3(
            pooled
        )

    return pool.time_out_proj(
        pool.dropout(
            pooled
        )
    )


def global_reconstruction(
    pool,
    global_freq,
    gate,
    mask,
):
    enhanced = (
        global_freq
        * gate.unsqueeze(1)
    )

    time_tokens = pool._back_to_time(
        enhanced,
        seq_len=mask.size(1),
    )

    embedding = finalize_embedding(
        pool,
        time_tokens,
        mask,
    )

    return embedding, time_tokens


def local_reconstruction(
    pool,
    local_freq,
    gate,
    meta,
    mask,
):
    enhanced = (
        local_freq
        * gate.unsqueeze(1)
    )

    time_tokens = (
        pool._back_from_local_frequency(
            enhanced,
            meta,
        )
    )

    time_tokens = (
        time_tokens
        * mask.unsqueeze(-1).to(
            time_tokens.dtype
        )
    )

    embedding = finalize_embedding(
        pool,
        time_tokens,
        mask,
    )

    return embedding, time_tokens


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        choices=["FLaG", "E12"],
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help=(
            "Evaluation batch size. For exact replay of the Sprint "
            "checkpoints this must match the training eval_batch_size "
            "(64), because global FLaG is sensitive to dynamic-padding "
            "FFT length."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        default=str(
            ROOT
            / "outputs/text/sprintduplicatequestions/probes"
            / "global_local_2x2"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    ckpt_dir = checkpoint_dir(
        args.source,
        args.seed,
    )

    with open(
        ckpt_dir / "config.json"
    ) as f:
        source_cfg = json.load(f)

    expected_eval_batch_size = int(
        source_cfg.get(
            "eval_batch_size",
            64,
        )
    )

    if (
        args.limit is None
        and args.batch_size
        != expected_eval_batch_size
    ):
        raise ValueError(
            "Full Sprint probe must replay the checkpoint's "
            "evaluation batch convention exactly. "
            f"Got --batch_size={args.batch_size}, "
            f"but config eval_batch_size="
            f"{expected_eval_batch_size}. "
            "Global FLaG depends on dynamic-padding FFT length."
        )

    print("device:", device)
    print("source:", args.source)
    print("seed:", args.seed)
    print("checkpoint:", ckpt_dir)
    print("split:", args.split)

    source_model = build_source_model(
        args.source,
        ckpt_dir,
        device,
    )

    d_model = (
        source_model
        .backbone
        .config
        .hidden_size
    )

    global_pool, local_pool = (
        make_operator_pools(
            source_model.pool,
            d_model,
            device,
        )
    )

    print("\nSource pool settings:")
    print(
        "dropout:",
        source_model.pool.dropout.p,
    )
    print(
        "post_pool_norm:",
        source_model.pool.post_pool_norm,
    )
    print(
        "time_pool:",
        source_model.pool.time_pool,
    )

    raw = load_from_disk(
        str(DATA_PATH)
    )

    ds = raw[args.split]

    n_total = len(ds)

    if args.limit is not None:
        n_total = min(
            n_total,
            args.limit,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    rows = []

    max_gg_manual_diff = 0.0
    max_ll_manual_diff = 0.0
    max_source_native_diff = 0.0

    with torch.inference_mode():

        for start in range(
            0,
            n_total,
            args.batch_size,
        ):
            end = min(
                start + args.batch_size,
                n_total,
            )

            batch = ds[start:end]

            s1 = list(
                batch["sentence1"]
            )
            s2 = list(
                batch["sentence2"]
            )

            true = np.asarray(
                batch["label"],
                dtype=np.int64,
            )

            B = len(s1)

            tokens = tokenizer(
                s1 + s2,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            tokens = {
                k: v.to(device)
                for k, v in tokens.items()
            }

            # Frozen backbone, exactly as Sprint training.
            hidden = (
                source_model
                .backbone(
                    **tokens
                )
                .last_hidden_state
            )

            mask = tokens[
                "attention_mask"
            ]

            # ------------------------------------------------
            # Build G/L frequency representations
            # ------------------------------------------------

            g_freq = build_global_frequency(
                global_pool,
                hidden,
                mask,
            )

            (
                l_freq,
                l_freq_mask,
                l_meta,
            ) = (
                local_pool
                ._build_local_frequency_tokens(
                    hidden,
                    mask,
                )
            )

            # ------------------------------------------------
            # Generate gates
            # ------------------------------------------------

            (
                gate_g_raw,
                gate_g,
            ) = global_gate(
                global_pool,
                g_freq,
            )

            (
                gate_l_raw,
                gate_l,
            ) = local_gate(
                local_pool,
                l_freq,
                l_freq_mask,
                l_meta,
            )

            # ------------------------------------------------
            # 2 x 2 operator decomposition
            #
            # first letter: gate observation G/L
            # second letter: reconstruction G/L
            # ------------------------------------------------

            emb_GG, _ = global_reconstruction(
                global_pool,
                g_freq,
                gate_g,
                mask,
            )

            emb_LG, _ = global_reconstruction(
                global_pool,
                g_freq,
                gate_l,
                mask,
            )

            emb_GL, _ = local_reconstruction(
                local_pool,
                l_freq,
                gate_g,
                l_meta,
                mask,
            )

            emb_LL, _ = local_reconstruction(
                local_pool,
                l_freq,
                gate_l,
                l_meta,
                mask,
            )

            # ------------------------------------------------
            # Exact native-forward sanity
            # ------------------------------------------------

            native_global = global_pool(
                hidden,
                attention_mask=mask,
            )

            native_local = local_pool(
                hidden,
                attention_mask=mask,
            )

            gg_diff = (
                emb_GG
                - native_global
            ).abs().max().item()

            ll_diff = (
                emb_LL
                - native_local
            ).abs().max().item()

            max_gg_manual_diff = max(
                max_gg_manual_diff,
                gg_diff,
            )

            max_ll_manual_diff = max(
                max_ll_manual_diff,
                ll_diff,
            )

            if gg_diff > 2e-5:
                raise RuntimeError(
                    "GG manual/native mismatch: "
                    f"{gg_diff}"
                )

            if ll_diff > 2e-5:
                raise RuntimeError(
                    "LL manual/native mismatch: "
                    f"{ll_diff}"
                )

            native_source = source_model.pool(
                hidden,
                attention_mask=mask,
            )

            source_ref = (
                native_global
                if args.source == "FLaG"
                else native_local
            )

            src_diff = (
                native_source
                - source_ref
            ).abs().max().item()

            max_source_native_diff = max(
                max_source_native_diff,
                src_diff,
            )

            if src_diff > 2e-5:
                raise RuntimeError(
                    "Source/native operator mismatch: "
                    f"{src_diff}"
                )

            # ------------------------------------------------
            # Pair cosine predictions
            # ------------------------------------------------

            pred_GG = pair_prediction(
                emb_GG,
                B,
            )

            pred_LG = pair_prediction(
                emb_LG,
                B,
            )

            pred_GL = pair_prediction(
                emb_GL,
                B,
            )

            pred_LL = pair_prediction(
                emb_LL,
                B,
            )

            # ------------------------------------------------
            # Gate diagnostics
            # ------------------------------------------------

            gate_mae = (
                gate_l_raw
                - gate_g_raw
            ).abs().mean(dim=1)

            gate_cos = F.cosine_similarity(
                gate_g_raw,
                gate_l_raw,
                dim=-1,
            )

            for i in range(B):
                j1 = i
                j2 = B + i

                rows.append({
                    "pair_id": start + i,
                    "true": int(true[i]),
                    "batch_T": int(
                        hidden.size(1)
                    ),
                    "len1": int(
                        mask[j1].sum().item()
                    ),
                    "len2": int(
                        mask[j2].sum().item()
                    ),
                    "GG": float(
                        pred_GG[i].cpu()
                    ),
                    "LG": float(
                        pred_LG[i].cpu()
                    ),
                    "GL": float(
                        pred_GL[i].cpu()
                    ),
                    "LL": float(
                        pred_LL[i].cpu()
                    ),
                    "gate_G_vs_L_mae": float(
                        (
                            gate_mae[j1]
                            + gate_mae[j2]
                        ).item()
                        / 2.0
                    ),
                    "gate_G_vs_L_cos": float(
                        (
                            gate_cos[j1]
                            + gate_cos[j2]
                        ).item()
                        / 2.0
                    ),
                })

            if (
                (start // args.batch_size) % 100 == 0
                or end == n_total
            ):
                print(
                    f"processed {end}/{n_total}"
                )

    df = pd.DataFrame(rows)

    out_dir = (
        Path(args.output_dir)
        / (
            f"{args.source}_"
            f"seed{args.seed}_"
            f"{args.split}"
        )
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out_dir / "pair_results.csv",
        index=False,
    )

    # ========================================================
    # Sanity
    # ========================================================

    print()
    print("=" * 90)
    print("SANITY")
    print("=" * 90)

    print(
        "max |manual GG - native global| =",
        f"{max_gg_manual_diff:.10g}",
    )

    print(
        "max |manual LL - native local|  =",
        f"{max_ll_manual_diff:.10g}",
    )

    print(
        "max |source model - source op|  =",
        f"{max_source_native_diff:.10g}",
    )

    # ========================================================
    # AP
    # ========================================================

    labels = df["true"].to_numpy()

    ap = {}

    print()
    print("=" * 90)
    print("AVERAGE PRECISION")
    print("=" * 90)

    for mode in ["GG", "LG", "GL", "LL"]:
        value = average_precision(
            df[mode],
            labels,
        )

        ap[mode] = value

        print(
            f"{mode}: {value:.9f}"
        )

    source_mode = (
        "GG"
        if args.source == "FLaG"
        else "LL"
    )

    with open(
        ckpt_dir / "metrics.json"
    ) as f:
        training_metrics = json.load(f)

    if args.split == "validation":
        expected_source_ap = float(
            training_metrics[
                "final_val_average_precision"
            ]
        )
    else:
        expected_source_ap = float(
            training_metrics[
                "test_average_precision"
            ]
        )

    source_ap_diff = abs(
        ap[source_mode]
        - expected_source_ap
    )

    print()
    print(
        "source mode:",
        source_mode,
    )
    print(
        "expected source AP:",
        f"{expected_source_ap:.9f}",
    )
    print(
        "probe source AP:",
        f"{ap[source_mode]:.9f}",
    )
    print(
        "|AP diff|:",
        f"{source_ap_diff:.10g}",
    )

    if (
        args.limit is None
        and source_ap_diff > 2e-5
    ):
        raise RuntimeError(
            "Probe source AP does not reproduce "
            "training metrics: "
            f"{source_ap_diff}"
        )

    # ========================================================
    # Prediction-level decomposition
    # ========================================================

    gg = df["GG"].to_numpy()
    lg = df["LG"].to_numpy()
    gl = df["GL"].to_numpy()
    ll = df["LL"].to_numpy()

    obs_effect = lg - gg
    recon_effect = gl - gg
    total_effect = ll - gg
    interaction = ll - lg - gl + gg

    print()
    print("=" * 90)
    print("PREDICTION-LEVEL FACTORIAL DECOMPOSITION")
    print("=" * 90)

    def effect_stats(name, x):
        stats = {
            "mean_abs": float(
                np.mean(np.abs(x))
            ),
            "median_abs": float(
                np.median(np.abs(x))
            ),
            "max_abs": float(
                np.max(np.abs(x))
            ),
            "mean_signed": float(
                np.mean(x)
            ),
        }

        print(
            f"{name:24s} "
            f"mean|effect|={stats['mean_abs']:.9f}  "
            f"median={stats['median_abs']:.9f}  "
            f"max={stats['max_abs']:.9f}  "
            f"mean_signed={stats['mean_signed']:+.9f}"
        )

        return stats

    obs_stats = effect_stats(
        "gate observation LG-GG",
        obs_effect,
    )

    recon_stats = effect_stats(
        "reconstruction GL-GG",
        recon_effect,
    )

    total_stats = effect_stats(
        "total local LL-GG",
        total_effect,
    )

    interaction_stats = effect_stats(
        "interaction",
        interaction,
    )

    corr_obs_total = safe_corr(
        obs_effect,
        total_effect,
    )

    corr_recon_total = safe_corr(
        recon_effect,
        total_effect,
    )

    print()
    print(
        "corr(LG-GG, LL-GG) =",
        f"{corr_obs_total:.9f}",
    )

    print(
        "corr(GL-GG, LL-GG) =",
        f"{corr_recon_total:.9f}",
    )

    gate_mae_mean = float(
        df["gate_G_vs_L_mae"].mean()
    )

    gate_cos_mean = float(
        df["gate_G_vs_L_cos"].mean()
    )

    print()
    print("=" * 90)
    print("GLOBAL vs LOCAL GATE")
    print("=" * 90)

    print(
        "mean gate MAE =",
        f"{gate_mae_mean:.9f}",
    )

    print(
        "mean gate cosine =",
        f"{gate_cos_mean:.9f}",
    )

    # AP changes from operator substitutions.
    ap_effects = {
        "LG_minus_GG":
            float(ap["LG"] - ap["GG"]),
        "GL_minus_GG":
            float(ap["GL"] - ap["GG"]),
        "LL_minus_GG":
            float(ap["LL"] - ap["GG"]),
    }

    print()
    print("=" * 90)
    print("AP OPERATOR EFFECTS")
    print("=" * 90)

    for key, value in ap_effects.items():
        print(
            f"{key}: {value:+.9f}"
        )

    summary = {
        "source": args.source,
        "seed": args.seed,
        "split": args.split,
        "n_pairs": int(len(df)),
        "source_pool": {
            "dropout": float(
                source_model.pool.dropout.p
            ),
            "post_pool_norm": bool(
                source_model.pool.post_pool_norm
            ),
            "time_pool":
                source_model.pool.time_pool,
        },
        "sanity": {
            "max_gg_manual_diff":
                max_gg_manual_diff,
            "max_ll_manual_diff":
                max_ll_manual_diff,
            "max_source_native_diff":
                max_source_native_diff,
            "source_mode":
                source_mode,
            "expected_source_ap":
                expected_source_ap,
            "probe_source_ap":
                ap[source_mode],
            "source_ap_abs_diff":
                source_ap_diff,
        },
        "average_precision": ap,
        "ap_effects": ap_effects,
        "prediction_effects": {
            "observation": obs_stats,
            "reconstruction": recon_stats,
            "total_local": total_stats,
            "interaction":
                interaction_stats,
            "corr_observation_total":
                corr_obs_total,
            "corr_reconstruction_total":
                corr_recon_total,
        },
        "gate": {
            "mean_mae":
                gate_mae_mean,
            "mean_cosine":
                gate_cos_mean,
        },
    }

    with open(
        out_dir / "summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        "Saved to:",
        out_dir,
    )


if __name__ == "__main__":
    main()
