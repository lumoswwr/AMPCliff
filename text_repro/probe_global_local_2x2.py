import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import load_from_disk
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

from text_repro.train_sts import SentenceEncoder

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
    ROOT / "data/text/stsbenchmark"
)


# ============================================================
# Basic utilities
# ============================================================

def metrics(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    return {
        "spearman": float(
            spearmanr(pred, true)[0]
        ),
        "pearson": float(
            pearsonr(pred, true)[0]
        ),
    }


def get_split(ds, name):
    if hasattr(ds, "keys"):
        return ds[name]

    if name != "test":
        raise RuntimeError(
            "Dataset object has no split mapping."
        )

    return ds


def detect_columns(ds):
    cols = list(ds.column_names)

    def choose(candidates):
        for x in candidates:
            if x in cols:
                return x

        raise RuntimeError(
            f"Could not detect column from {candidates}. "
            f"Available columns: {cols}"
        )

    return (
        choose([
            "sentence1",
            "sentence_1",
            "sent1",
            "text1",
        ]),
        choose([
            "sentence2",
            "sentence_2",
            "sent2",
            "text2",
        ]),
        choose([
            "score",
            "label",
            "labels",
            "similarity_score",
        ]),
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


# ============================================================
# Find checkpoints
# ============================================================

def is_e12_config(cfg):
    try:
        pooling = cfg.get(
            "pooling"
        )

        win = int(
            cfg.get(
                "stft_win_length",
                -1,
            )
        )

        hop = int(
            cfg.get(
                "stft_hop_length",
                -1,
            )
        )

        window = cfg.get(
            "stft_window_type",
            "rect",
        )

        center = bool(
            cfg.get(
                "stft_center",
                False,
            )
        )

        return (
            pooling == "STFT_FLaG"
            and win == 16
            and hop == 16
            and window == "rect"
            and not center
        )

    except Exception:
        return False


def find_checkpoint_dir(
    source,
    seed,
    explicit_dir=None,
):
    if explicit_dir is not None:
        p = Path(
            explicit_dir
        )

        if not (
            p / "best_model.pt"
        ).exists():
            raise FileNotFoundError(
                p / "best_model.pt"
            )

        return p

    if source == "FLaG":
        p = (
            ROOT
            / "outputs/text/stsbenchmark"
            / "FLaG"
            / f"seed_{seed}"
        )

        if (
            p / "best_model.pt"
        ).exists():
            return p

        raise FileNotFoundError(
            p / "best_model.pt"
        )

    exp_root = (
        ROOT
        / "outputs/text/stsbenchmark"
        / "experiments"
    )

    candidates = []

    for cfg_path in exp_root.glob(
        f"*/seed_{seed}/config.json"
    ):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)

            if is_e12_config(cfg):
                candidates.append(
                    cfg_path.parent
                )

        except Exception:
            continue

    if len(candidates) == 1:
        return candidates[0]

    e12_named = [
        p
        for p in candidates
        if "e12" in str(p).lower()
    ]

    if len(e12_named) == 1:
        return e12_named[0]

    print(
        "E12 candidates:"
    )

    for p in candidates:
        print(
            "  ",
            p,
        )

    raise RuntimeError(
        "Could not uniquely identify E12 run. "
        "Pass --checkpoint_dir explicitly."
    )


# ============================================================
# Construct source model
# ============================================================

def build_source_model(
    source,
    checkpoint_dir,
    device,
):
    if source == "FLaG":
        model = SentenceEncoder(
            str(MODEL_PATH),
            "FLaG",
            fixed_fft_length=None,
        )

    elif source == "E12":
        model = SentenceEncoder(
            str(MODEL_PATH),
            "STFT_FLaG",
            stft_win_length=16,
            stft_hop_length=16,
            stft_window_type="rect",
            stft_center=False,
            fixed_fft_length=None,
        )

    else:
        raise ValueError(
            source
        )

    ckpt = torch.load(
        checkpoint_dir
        / "best_model.pt",
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    return model


# ============================================================
# Create global/local copies with IDENTICAL weights
# ============================================================

def make_operator_pools(
    source_pool,
    d_model,
    device,
):
    global_pool = (
        FFTLatentAttentionGatePooling(
            d_model=d_model,
            num_latents=8,
            num_heads=4,
            dropout=0.0,
            time_pool="max",
            gate_residual=True,
            eps=1e-6,
            use_gate=True,
            use_latent=True,
            post_pool_norm=True,
            window_type=None,
            fixed_fft_length=None,
        )
    )

    local_pool = (
        STFTLatentAttentionGatePooling(
            d_model=d_model,
            win_length=16,
            hop_length=16,
            num_latents=8,
            num_heads=4,
            dropout=0.0,
            time_pool="max",
            gate_residual=True,
            eps=1e-6,
            use_gate=True,
            use_latent=True,
            post_pool_norm=True,
            use_frame_positional_encoding=False,
            max_frame_positions=32,
            local_window_type="rect",
            center_frames=False,
            fixed_fft_length=None,
        )
    )

    state = (
        source_pool.state_dict()
    )

    global_pool.load_state_dict(
        state,
        strict=True,
    )

    local_pool.load_state_dict(
        state,
        strict=True,
    )

    global_pool = (
        global_pool.to(device)
    )

    local_pool = (
        local_pool.to(device)
    )

    global_pool.eval()
    local_pool.eval()

    return (
        global_pool,
        local_pool,
    )


# ============================================================
# Gate helpers
# ============================================================

def gate_scale(
    pool,
    latent_out,
):
    raw = torch.sigmoid(
        pool.freq_gate(
            latent_out.mean(
                dim=1
            )
        )
    )

    if pool.gate_residual:
        scale = 1.0 + raw
    else:
        scale = raw

    return (
        raw,
        scale,
    )


def build_global_frequency(
    pool,
    hidden,
    mask,
):
    # This preserves the current global FLaG masking logic.
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
        [
            spec.real,
            spec.imag,
        ],
        dim=-1,
    )


def global_gate(
    pool,
    global_freq,
):
    latent = (
        pool._latent_pool_in_frequency(
            global_freq
        )
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

    latent = (
        pool._latent_pool_in_frequency(
            attention_tokens,
            freq_attention_mask=(
                freq_mask
            ),
        )
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
        pooled = (
            masked_mean_pooling(
                time_tokens,
                mask,
                eps=pool.eps,
            )
        )

    elif pool.time_pool == "max":
        pooled = (
            masked_max_pooling(
                time_tokens,
                mask,
            )
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

    T = mask.size(1)

    time_tokens = (
        pool._back_to_time(
            enhanced,
            seq_len=T,
        )
    )

    embedding = (
        finalize_embedding(
            pool,
            time_tokens,
            mask,
        )
    )

    return (
        embedding,
        time_tokens,
    )


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

    # Match native STFT forward exactly.
    time_tokens = (
        time_tokens
        * mask.unsqueeze(-1).to(
            time_tokens.dtype
        )
    )

    embedding = (
        finalize_embedding(
            pool,
            time_tokens,
            mask,
        )
    )

    return (
        embedding,
        time_tokens,
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        choices=[
            "FLaG",
            "E12",
        ],
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--split",
        choices=[
            "validation",
            "test",
        ],
        default="validation",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--checkpoint_dir",
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        default=str(
            ROOT
            / "outputs/text/stsbenchmark/probes"
            / "P2_global_local_2x2"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device =",
        device,
    )

    ckpt_dir = (
        find_checkpoint_dir(
            args.source,
            args.seed,
            args.checkpoint_dir,
        )
    )

    print(
        "source =",
        args.source,
    )

    print(
        "checkpoint =",
        ckpt_dir,
    )

    source_model = (
        build_source_model(
            args.source,
            ckpt_dir,
            device,
        )
    )

    d_model = (
        source_model
        .backbone
        .config
        .hidden_size
    )

    (
        global_pool,
        local_pool,
    ) = make_operator_pools(
        source_model.pool,
        d_model,
        device,
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    raw = load_from_disk(
        str(DATA_PATH)
    )

    ds = get_split(
        raw,
        args.split,
    )

    s1_col, s2_col, y_col = (
        detect_columns(ds)
    )

    n_total = len(ds)

    if args.limit is not None:
        n_total = min(
            n_total,
            args.limit,
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
        )
    )

    rows = []

    max_gg_manual_diff = 0.0
    max_ll_manual_diff = 0.0
    max_source_native_diff = 0.0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with torch.inference_mode():

        for start in range(
            0,
            n_total,
            args.batch_size,
        ):

            end = min(
                start
                + args.batch_size,
                n_total,
            )

            batch = ds[
                start:end
            ]

            s1 = list(
                batch[s1_col]
            )

            s2 = list(
                batch[s2_col]
            )

            true = np.asarray(
                batch[y_col],
                dtype=float,
            )

            B = len(s1)

            # Same joint dynamic padding convention as training.
            tokens = tokenizer(
                s1 + s2,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            tokens = {
                k: v.to(device)
                for k, v
                in tokens.items()
            }

            hidden = (
                source_model
                .backbone(
                    **tokens
                )
                .last_hidden_state
            )

            mask = (
                tokens[
                    "attention_mask"
                ]
            )

            # =================================================
            # Build both spectral representations
            # =================================================

            g_freq = (
                build_global_frequency(
                    global_pool,
                    hidden,
                    mask,
                )
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

            # =================================================
            # Generate gate from G or L spectrum
            # =================================================

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

            # =================================================
            # 2 x 2
            #
            # first letter  = gate observation
            # second letter = reconstruction
            # =================================================

            # GG:
            # global gate, global reconstruction
            emb_GG, _ = (
                global_reconstruction(
                    global_pool,
                    g_freq,
                    gate_g,
                    mask,
                )
            )

            # LG:
            # LOCAL gate, GLOBAL reconstruction
            emb_LG, _ = (
                global_reconstruction(
                    global_pool,
                    g_freq,
                    gate_l,
                    mask,
                )
            )

            # GL:
            # GLOBAL gate, LOCAL reconstruction
            emb_GL, _ = (
                local_reconstruction(
                    local_pool,
                    l_freq,
                    gate_g,
                    l_meta,
                    mask,
                )
            )

            # LL:
            # local gate, local reconstruction
            emb_LL, _ = (
                local_reconstruction(
                    local_pool,
                    l_freq,
                    gate_l,
                    l_meta,
                    mask,
                )
            )

            # =================================================
            # Native-forward sanity checks
            # =================================================

            native_global = (
                global_pool(
                    hidden,
                    attention_mask=mask,
                )
            )

            native_local = (
                local_pool(
                    hidden,
                    attention_mask=mask,
                )
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
                    "GG manual reconstruction does not "
                    f"match native global: {gg_diff}"
                )

            if ll_diff > 2e-5:
                raise RuntimeError(
                    "LL manual reconstruction does not "
                    f"match native local: {ll_diff}"
                )

            # Source checkpoint must also reproduce its
            # corresponding native operator.
            native_source = (
                source_model.pool(
                    hidden,
                    attention_mask=mask,
                )
            )

            if args.source == "FLaG":
                source_ref = native_global
            else:
                source_ref = native_local

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
                    "Source-model replay failed: "
                    f"{src_diff}"
                )

            # =================================================
            # Predictions
            # =================================================

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

            # =================================================
            # Gate difference diagnostics
            # =================================================

            gate_mae = (
                gate_l_raw
                - gate_g_raw
            ).abs().mean(
                dim=1
            )

            gate_cos = (
                F.cosine_similarity(
                    gate_g_raw,
                    gate_l_raw,
                    dim=-1,
                )
            )

            for i in range(B):

                j1 = i
                j2 = B + i

                pair_gate_mae = float(
                    (
                        gate_mae[j1]
                        + gate_mae[j2]
                    ).item()
                    / 2.0
                )

                pair_gate_cos = float(
                    (
                        gate_cos[j1]
                        + gate_cos[j2]
                    ).item()
                    / 2.0
                )

                rows.append({
                    "pair_id":
                        start + i,

                    "true":
                        float(
                            true[i]
                        ),

                    "batch_T":
                        int(
                            hidden.size(1)
                        ),

                    "len1":
                        int(
                            mask[j1]
                            .sum()
                            .item()
                        ),

                    "len2":
                        int(
                            mask[j2]
                            .sum()
                            .item()
                        ),

                    "GG":
                        float(
                            pred_GG[i]
                            .cpu()
                        ),

                    "LG":
                        float(
                            pred_LG[i]
                            .cpu()
                        ),

                    "GL":
                        float(
                            pred_GL[i]
                            .cpu()
                        ),

                    "LL":
                        float(
                            pred_LL[i]
                            .cpu()
                        ),

                    "gate_G_vs_L_mae":
                        pair_gate_mae,

                    "gate_G_vs_L_cos":
                        pair_gate_cos,
                })

    # ========================================================
    # Summary
    # ========================================================

    df = pd.DataFrame(
        rows
    )

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
        out_dir
        / "pair_results.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("SANITY")
    print("=" * 100)

    print(
        "max |manual GG - native global| =",
        f"{max_gg_manual_diff:.10g}",
    )

    print(
        "max |manual LL - native local|  =",
        f"{max_ll_manual_diff:.10g}",
    )

    print(
        "max |source model - source op|   =",
        f"{max_source_native_diff:.10g}",
    )

    print()
    print("=" * 100)
    print("CORRELATIONS")
    print("=" * 100)

    summary = {
        "source":
            args.source,

        "seed":
            args.seed,

        "split":
            args.split,

        "n_pairs":
            len(df),

        "metrics": {},
    }

    for mode in [
        "GG",
        "LG",
        "GL",
        "LL",
    ]:
        m = metrics(
            df[mode],
            df["true"],
        )

        summary[
            "metrics"
        ][mode] = m

        print(
            f"{mode:3s}  "
            f"Spearman={m['spearman']:.9f}  "
            f"Pearson={m['pearson']:.9f}"
        )

    # ========================================================
    # Prediction-level decomposition
    # ========================================================

    gg = df["GG"].to_numpy()
    lg = df["LG"].to_numpy()
    gl = df["GL"].to_numpy()
    ll = df["LL"].to_numpy()

    # Local gate observation effect,
    # holding GLOBAL reconstruction fixed.
    obs_effect = (
        lg - gg
    )

    # Local reconstruction effect,
    # holding GLOBAL gate fixed.
    recon_effect = (
        gl - gg
    )

    # Total local operator effect.
    total_effect = (
        ll - gg
    )

    # Factorial interaction at prediction level.
    interaction = (
        ll - lg - gl + gg
    )

    print()
    print("=" * 100)
    print("PREDICTION-LEVEL FACTORIAL DECOMPOSITION")
    print("=" * 100)

    def show_effect(
        name,
        x,
    ):
        print(
            f"{name:24s} "
            f"mean|effect|={np.mean(np.abs(x)):.9f}  "
            f"median={np.median(np.abs(x)):.9f}  "
            f"max={np.max(np.abs(x)):.9f}  "
            f"mean_signed={np.mean(x):+.9f}"
        )

    show_effect(
        "gate observation LG-GG",
        obs_effect,
    )

    show_effect(
        "reconstruction GL-GG",
        recon_effect,
    )

    show_effect(
        "total local LL-GG",
        total_effect,
    )

    show_effect(
        "interaction",
        interaction,
    )

    print()
    print("=" * 100)
    print("WHICH COMPONENT TRACKS LL-GG?")
    print("=" * 100)

    def safe_corr(a, b):
        if (
            np.std(a) == 0
            or np.std(b) == 0
        ):
            return float("nan")

        return float(
            np.corrcoef(
                a,
                b,
            )[0, 1]
        )

    corr_obs_total = (
        safe_corr(
            obs_effect,
            total_effect,
        )
    )

    corr_recon_total = (
        safe_corr(
            recon_effect,
            total_effect,
        )
    )

    print(
        "corr(LG-GG, LL-GG) =",
        f"{corr_obs_total:.9f}",
    )

    print(
        "corr(GL-GG, LL-GG) =",
        f"{corr_recon_total:.9f}",
    )

    print()
    print("=" * 100)
    print("GLOBAL vs LOCAL GATE")
    print("=" * 100)

    print(
        "mean gate MAE =",
        f"{df['gate_G_vs_L_mae'].mean():.9f}",
    )

    print(
        "mean gate cosine =",
        f"{df['gate_G_vs_L_cos'].mean():.9f}",
    )

    summary["prediction_effects"] = {
        "mean_abs_observation":
            float(
                np.mean(
                    np.abs(
                        obs_effect
                    )
                )
            ),

        "mean_abs_reconstruction":
            float(
                np.mean(
                    np.abs(
                        recon_effect
                    )
                )
            ),

        "mean_abs_total":
            float(
                np.mean(
                    np.abs(
                        total_effect
                    )
                )
            ),

        "mean_abs_interaction":
            float(
                np.mean(
                    np.abs(
                        interaction
                    )
                )
            ),

        "corr_observation_total":
            corr_obs_total,

        "corr_reconstruction_total":
            corr_recon_total,

        "gate_mae":
            float(
                df[
                    "gate_G_vs_L_mae"
                ].mean()
            ),

        "gate_cosine":
            float(
                df[
                    "gate_G_vs_L_cos"
                ].mean()
            ),
    }

    with open(
        out_dir
        / "summary.json",
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
