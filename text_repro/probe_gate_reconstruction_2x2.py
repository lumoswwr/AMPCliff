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
from AMPCliff.factory.pooling.llm_pooling_dropin import (
    masked_max_pooling,
    masked_mean_pooling,
)


ROOT = Path(
    "/home/data/home/wwr_lumos/AMPCliff"
)

DEFAULT_MODEL = Path(
    "/home/data/home/wwr_lumos/models/roberta-base"
)

DEFAULT_DATA = (
    ROOT / "data/text/stsbenchmark"
)


# ============================================================
# Utilities
# ============================================================

def correlation_metrics(pred, true):
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


def get_split(ds, split):
    if hasattr(ds, "keys"):
        if split not in ds:
            raise KeyError(
                f"Split {split!r} not found. "
                f"Available: {list(ds.keys())}"
            )
        return ds[split]

    if split != "test":
        raise RuntimeError(
            "load_from_disk returned a single Dataset, "
            f"but requested split={split!r}"
        )

    return ds


def detect_columns(ds):
    cols = list(ds.column_names)

    s1_candidates = [
        "sentence1",
        "sentence_1",
        "sent1",
        "text1",
    ]

    s2_candidates = [
        "sentence2",
        "sentence_2",
        "sent2",
        "text2",
    ]

    y_candidates = [
        "score",
        "label",
        "labels",
        "similarity_score",
    ]

    def pick(candidates, name):
        for c in candidates:
            if c in cols:
                return c

        raise RuntimeError(
            f"Could not detect {name} column. "
            f"Columns = {cols}"
        )

    return (
        pick(s1_candidates, "sentence1"),
        pick(s2_candidates, "sentence2"),
        pick(y_candidates, "score"),
    )


# ============================================================
# Frequency / gate decomposition
# ============================================================

def masked_input(
    pool,
    hidden,
    mask,
):
    # Uses the exact current FLaG preprocessing.
    return pool._apply_input_window(
        hidden,
        mask,
    )


def frequency_tokens_at_n(
    pool,
    hidden,
    mask,
    n_fft,
):
    x = masked_input(
        pool,
        hidden,
        mask,
    )

    spec = torch.fft.rfft(
        x,
        n=n_fft,
        dim=1,
    )

    return torch.cat(
        [
            spec.real,
            spec.imag,
        ],
        dim=-1,
    )


def gate_from_n(
    pool,
    hidden,
    mask,
    n_gate,
):
    freq = frequency_tokens_at_n(
        pool,
        hidden,
        mask,
        n_gate,
    )

    if not pool.use_latent:
        raise RuntimeError(
            "This probe assumes use_latent=True."
        )

    latent = (
        pool._latent_pool_in_frequency(
            freq
        )
    )

    gate_raw = torch.sigmoid(
        pool.freq_gate(
            latent.mean(dim=1)
        )
    )

    if pool.gate_residual:
        gate_scale = 1.0 + gate_raw
    else:
        gate_scale = gate_raw

    return gate_raw, gate_scale


def symmetrize_gate_scale(
    gate_scale,
    d_model,
):
    """
    Force the same multiplier on real and imaginary
    parts of each hidden channel.

    If original multipliers are a and b:

        shared = (a + b) / 2

    so beta = (a - b) / 2 becomes exactly zero.
    """

    a = gate_scale[:, :d_model]
    b = gate_scale[:, d_model:]

    shared = (
        a + b
    ) / 2.0

    return torch.cat(
        [
            shared,
            shared,
        ],
        dim=-1,
    )


def reconstruct_at_n(
    pool,
    hidden,
    mask,
    gate_scale,
    n_recon,
):
    """
    Important:
    recompute the COMPLETE spectrum at n_recon,
    then apply the gate generated at n_gate.

    We never reuse an Ng spectrum directly for an Nr iFFT.
    """

    B, T, D = hidden.shape

    if n_recon < T:
        raise ValueError(
            f"n_recon={n_recon} < tensor T={T}"
        )

    freq = frequency_tokens_at_n(
        pool,
        hidden,
        mask,
        n_recon,
    )

    enhanced = (
        freq
        * gate_scale.unsqueeze(1)
    )

    real, imag = enhanced.chunk(
        2,
        dim=-1,
    )

    spec = torch.complex(
        real,
        imag,
    )

    time_full = torch.fft.irfft(
        spec,
        n=n_recon,
        dim=1,
    )

    time_tokens = (
        time_full[:, :T, :]
    )

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
            f"Unexpected time_pool={pool.time_pool}"
        )

    if pool.post_pool_norm:
        pooled_for_projection = (
            pool.norm3(pooled)
        )
    else:
        pooled_for_projection = pooled

    embedding = pool.time_out_proj(
        pool.dropout(
            pooled_for_projection
        )
    )

    return embedding, time_tokens


# ============================================================
# Diagnostics
# ============================================================

def bos_max_fraction(
    time_tokens,
    mask,
):
    """
    Fraction of hidden channels whose max-pooled
    source position is BOS (position 0).
    """

    valid = mask.bool().unsqueeze(-1)

    x = time_tokens.masked_fill(
        ~valid,
        float("-inf"),
    )

    argmax_pos = x.argmax(
        dim=1
    )

    return (
        argmax_pos == 0
    ).float().mean(
        dim=1
    )


def reflection_ratio(
    pool,
    hidden,
    mask,
    gate_scale,
    n_recon,
):
    """
    For the given gate, use

        y_t = alpha*x_t + beta*x_{(-t) mod N}

    and measure RMS(reflection) / RMS(direct)
    on valid NON-BOS positions.

    This is a structural diagnostic, not a causal
    importance score.
    """

    B, T, D = hidden.shape

    x = masked_input(
        pool,
        hidden,
        mask,
    )

    if n_recon < T:
        raise ValueError(
            f"n_recon={n_recon} < T={T}"
        )

    if n_recon > T:
        x_n = F.pad(
            x,
            (
                0, 0,
                0, n_recon - T,
            ),
        )

        mask_n = F.pad(
            mask,
            (
                0,
                n_recon - T,
            ),
            value=0,
        )
    else:
        x_n = x
        mask_n = mask

    a = gate_scale[:, :D]
    b = gate_scale[:, D:]

    alpha = (
        a + b
    ) / 2.0

    beta = (
        a - b
    ) / 2.0

    idx = (
        -torch.arange(
            n_recon,
            device=x.device,
        )
    ) % n_recon

    reflected_x = x_n[
        :,
        idx,
        :,
    ]

    direct = (
        alpha.unsqueeze(1)
        * x_n
    )

    reflected = (
        beta.unsqueeze(1)
        * reflected_x
    )

    valid = mask_n.bool()

    # Exclude BOS at t=0.
    valid[:, 0] = False

    valid3 = valid.unsqueeze(-1)

    direct_sq = (
        direct.pow(2)
        * valid3
    ).sum(
        dim=(1, 2)
    )

    refl_sq = (
        reflected.pow(2)
        * valid3
    ).sum(
        dim=(1, 2)
    )

    denom = (
        valid3.sum(
            dim=(1, 2)
        )
        * D
    ).clamp(
        min=1
    )

    direct_rms = torch.sqrt(
        direct_sq / denom
    )

    refl_rms = torch.sqrt(
        refl_sq / denom
    )

    return (
        refl_rms
        / direct_rms.clamp(
            min=1e-12
        )
    )


def pair_cosine(
    embeddings,
    batch_size,
):
    z1 = embeddings[:batch_size]
    z2 = embeddings[batch_size:]

    return F.cosine_similarity(
        z1,
        z2,
        dim=-1,
    )


# ============================================================
# Main probe
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
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
        "--fixed_n",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--replay_only",
        action="store_true",
    )

    parser.add_argument(
        "--check_saved_predictions",
        action="store_true",
    )

    parser.add_argument(
        "--model_path",
        default=str(
            DEFAULT_MODEL
        ),
    )

    parser.add_argument(
        "--data_path",
        default=str(
            DEFAULT_DATA
        ),
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
            / "P1_gate_reconstruction_2x2"
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

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    if args.checkpoint_dir is None:
        ckpt_dir = (
            ROOT
            / "outputs/text/stsbenchmark"
            / "FLaG"
            / f"seed_{args.seed}"
        )
    else:
        ckpt_dir = Path(
            args.checkpoint_dir
        )

    ckpt_path = (
        ckpt_dir
        / "best_model.pt"
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(
            ckpt_path
        )

    # This probe intentionally uses the ORIGINAL
    # dynamic-FLaG checkpoint.
    model = SentenceEncoder(
        args.model_path,
        "FLaG",
        fixed_fft_length=None,
    ).to(device)

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    model.eval()

    pool = model.pool

    if pool.fixed_fft_length is not None:
        raise RuntimeError(
            "This probe must start from the "
            "original dynamic FLaG checkpoint."
        )

    print(
        "checkpoint =",
        ckpt_path,
    )

    print(
        "pool.fixed_fft_length =",
        pool.fixed_fft_length,
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    raw = load_from_disk(
        args.data_path
    )

    ds = get_split(
        raw,
        args.split,
    )

    s1_col, s2_col, y_col = (
        detect_columns(ds)
    )

    print(
        "columns:",
        s1_col,
        s2_col,
        y_col,
    )

    n_total = len(ds)

    if args.limit is not None:
        n_total = min(
            n_total,
            args.limit,
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model_path,
            local_files_only=True,
        )
    )

    rows = []

    max_native_tt_diff = 0.0

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

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

            batch = ds[
                start:end
            ]

            s1 = list(
                batch[s1_col]
            )

            s2 = list(
                batch[s2_col]
            )

            y = np.asarray(
                batch[y_col],
                dtype=float,
            )

            B = len(s1)

            # Joint tokenization makes both sentence sides
            # share the same dynamic batch T.
            texts = s1 + s2

            tokens = tokenizer(
                texts,
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

            outputs = model.backbone(
                **tokens
            )

            hidden = (
                outputs.last_hidden_state
            )

            mask = (
                tokens[
                    "attention_mask"
                ]
            )

            T = hidden.size(1)

            # Native current FLaG.
            native_embedding = pool(
                hidden,
                attention_mask=mask,
            )

            # ------------------------------------------------
            # Gate at dynamic T
            # ------------------------------------------------

            gate_T_raw, gate_T = (
                gate_from_n(
                    pool,
                    hidden,
                    mask,
                    T,
                )
            )

            # TT, our reimplementation of native FLaG.
            emb_TT, time_TT = (
                reconstruct_at_n(
                    pool,
                    hidden,
                    mask,
                    gate_T,
                    T,
                )
            )

            diff = (
                emb_TT
                - native_embedding
            ).abs().max().item()

            max_native_tt_diff = max(
                max_native_tt_diff,
                diff,
            )

            if diff > 2e-5:
                raise RuntimeError(
                    "TT does not reproduce native FLaG: "
                    f"max diff={diff}"
                )

            pred_TT = pair_cosine(
                emb_TT,
                B,
            )

            # ------------------------------------------------
            # SYM:
            # Same dynamic FFT length T and same learned gate,
            # but enforce identical real/imag multipliers.
            # This removes the circular-reflection term.
            # ------------------------------------------------

            gate_T_sym = symmetrize_gate_scale(
                gate_T,
                hidden.size(-1),
            )

            emb_SYM, time_SYM = (
                reconstruct_at_n(
                    pool,
                    hidden,
                    mask,
                    gate_T_sym,
                    T,
                )
            )

            pred_SYM = pair_cosine(
                emb_SYM,
                B,
            )

            if args.replay_only:

                for i in range(B):
                    rows.append({
                        "pair_id":
                            start + i,

                        "true":
                            float(y[i]),

                        "TT":
                            float(
                                pred_TT[i]
                                .cpu()
                            ),

                        "batch_T":
                            int(T),

                        "len1":
                            int(
                                mask[i]
                                .sum()
                                .item()
                            ),

                        "len2":
                            int(
                                mask[B + i]
                                .sum()
                                .item()
                            ),
                    })

                continue

            # ------------------------------------------------
            # Gate at fixed 128
            # ------------------------------------------------

            gate_128_raw, gate_128 = (
                gate_from_n(
                    pool,
                    hidden,
                    mask,
                    args.fixed_n,
                )
            )

            # T128:
            # gate from T, reconstruct at 128.
            emb_T128, time_T128 = (
                reconstruct_at_n(
                    pool,
                    hidden,
                    mask,
                    gate_T,
                    args.fixed_n,
                )
            )

            # 128T:
            # gate from 128, reconstruct at T.
            emb_128T, time_128T = (
                reconstruct_at_n(
                    pool,
                    hidden,
                    mask,
                    gate_128,
                    T,
                )
            )

            # 128128:
            # gate from 128, reconstruct at 128.
            emb_128128, time_128128 = (
                reconstruct_at_n(
                    pool,
                    hidden,
                    mask,
                    gate_128,
                    args.fixed_n,
                )
            )

            pred_T128 = pair_cosine(
                emb_T128,
                B,
            )

            pred_128T = pair_cosine(
                emb_128T,
                B,
            )

            pred_128128 = pair_cosine(
                emb_128128,
                B,
            )

            # ------------------------------------------------
            # Gate differences
            # ------------------------------------------------

            gate_mae = (
                gate_128_raw
                - gate_T_raw
            ).abs().mean(
                dim=1
            )

            gate_cos = (
                F.cosine_similarity(
                    gate_T_raw,
                    gate_128_raw,
                    dim=-1,
                )
            )

            # ------------------------------------------------
            # BOS max-source fractions
            # ------------------------------------------------

            bos_TT = bos_max_fraction(
                time_TT,
                mask,
            )

            bos_SYM = bos_max_fraction(
                time_SYM,
                mask,
            )

            bos_T128 = bos_max_fraction(
                time_T128,
                mask,
            )

            bos_128T = bos_max_fraction(
                time_128T,
                mask,
            )

            bos_128128 = bos_max_fraction(
                time_128128,
                mask,
            )

            # ------------------------------------------------
            # Reflection ratios
            # ------------------------------------------------

            refl_TT = reflection_ratio(
                pool,
                hidden,
                mask,
                gate_T,
                T,
            )

            refl_SYM = reflection_ratio(
                pool,
                hidden,
                mask,
                gate_T_sym,
                T,
            )

            refl_T128 = reflection_ratio(
                pool,
                hidden,
                mask,
                gate_T,
                args.fixed_n,
            )

            refl_128T = reflection_ratio(
                pool,
                hidden,
                mask,
                gate_128,
                T,
            )

            refl_128128 = reflection_ratio(
                pool,
                hidden,
                mask,
                gate_128,
                args.fixed_n,
            )

            # ------------------------------------------------
            # Save pair-level values
            # ------------------------------------------------

            for i in range(B):

                j1 = i
                j2 = B + i

                def pair_mean(v):
                    return float(
                        (
                            v[j1]
                            + v[j2]
                        ).item()
                        / 2.0
                    )

                rows.append({
                    "pair_id":
                        start + i,

                    "true":
                        float(y[i]),

                    "batch_T":
                        int(T),

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

                    "TT":
                        float(
                            pred_TT[i]
                            .cpu()
                        ),

                    "SYM":
                        float(
                            pred_SYM[i]
                            .cpu()
                        ),

                    "T128":
                        float(
                            pred_T128[i]
                            .cpu()
                        ),

                    "128T":
                        float(
                            pred_128T[i]
                            .cpu()
                        ),

                    "128128":
                        float(
                            pred_128128[i]
                            .cpu()
                        ),

                    "gate_T_to_128_mae":
                        pair_mean(
                            gate_mae
                        ),

                    "gate_T_to_128_cos":
                        pair_mean(
                            gate_cos
                        ),

                    "bosmax_TT":
                        pair_mean(
                            bos_TT
                        ),

                    "bosmax_SYM":
                        pair_mean(
                            bos_SYM
                        ),

                    "bosmax_T128":
                        pair_mean(
                            bos_T128
                        ),

                    "bosmax_128T":
                        pair_mean(
                            bos_128T
                        ),

                    "bosmax_128128":
                        pair_mean(
                            bos_128128
                        ),

                    "reflection_TT":
                        pair_mean(
                            refl_TT
                        ),

                    "reflection_SYM":
                        pair_mean(
                            refl_SYM
                        ),

                    "reflection_T128":
                        pair_mean(
                            refl_T128
                        ),

                    "reflection_128T":
                        pair_mean(
                            refl_128T
                        ),

                    "reflection_128128":
                        pair_mean(
                            refl_128128
                        ),
                })

    # ========================================================
    # Results
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    out_dir = (
        Path(args.output_dir)
        / f"FLaG_seed{args.seed}_{args.split}"
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
        "max |native FLaG - TT| =",
        f"{max_native_tt_diff:.10g}",
    )

    # --------------------------------------------------------
    # Replay check
    # --------------------------------------------------------

    if args.check_saved_predictions:

        if args.split != "test":
            raise RuntimeError(
                "--check_saved_predictions requires --split test"
            )

        saved_path = (
            ckpt_dir
            / "test_predictions.csv"
        )

        saved = pd.read_csv(
            saved_path
        ).iloc[
            :len(df)
        ]

        if len(saved) != len(df):
            raise RuntimeError(
                "Saved prediction row count mismatch."
            )

        replay_diff = np.max(
            np.abs(
                saved["prediction"]
                .to_numpy()
                - df["TT"].to_numpy()
            )
        )

        print(
            "max |saved prediction - TT| =",
            f"{replay_diff:.10g}",
        )

        if replay_diff > 2e-5:
            raise RuntimeError(
                "Replay does not match saved predictions."
            )

        print(
            "SAVED PREDICTION REPLAY PASSED"
        )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    modes = (
        ["TT"]
        if args.replay_only
        else [
            "TT",
            "SYM",
            "T128",
            "128T",
            "128128",
        ]
    )

    summary = {
        "seed":
            args.seed,

        "split":
            args.split,

        "n_pairs":
            len(df),

        "fixed_n":
            args.fixed_n,

        "max_native_tt_diff":
            max_native_tt_diff,

        "modes": {},
    }

    print()
    print("=" * 100)
    print("CORRELATIONS")
    print("=" * 100)

    for mode in modes:

        m = correlation_metrics(
            df[mode],
            df["true"],
        )

        summary[
            "modes"
        ][mode] = m

        print(
            f"{mode:8s} "
            f"Spearman={m['spearman']:.9f}  "
            f"Pearson={m['pearson']:.9f}"
        )

    if not args.replay_only:

        print()
        print("=" * 100)
        print("COUNTERFACTUAL PREDICTION DRIFT VS TT")
        print("=" * 100)

        for mode in [
            "SYM",
            "T128",
            "128T",
            "128128",
        ]:

            d = (
                df[mode]
                - df["TT"]
            ).to_numpy()

            stats = {
                "mean_abs":
                    float(
                        np.mean(
                            np.abs(d)
                        )
                    ),

                "median_abs":
                    float(
                        np.median(
                            np.abs(d)
                        )
                    ),

                "max_abs":
                    float(
                        np.max(
                            np.abs(d)
                        )
                    ),

                "mean_signed":
                    float(
                        np.mean(d)
                    ),
            }

            summary[
                f"drift_{mode}"
            ] = stats

            print(
                f"{mode:8s} "
                f"mean|Δpred|={stats['mean_abs']:.9f}  "
                f"median={stats['median_abs']:.9f}  "
                f"max={stats['max_abs']:.9f}  "
                f"meanΔ={stats['mean_signed']:+.9f}"
            )

        print()
        print("=" * 100)
        print("GATE OBSERVATION CHANGE: T -> 128")
        print("=" * 100)

        print(
            "mean gate MAE =",
            f"{df['gate_T_to_128_mae'].mean():.9f}"
        )

        print(
            "mean gate cosine =",
            f"{df['gate_T_to_128_cos'].mean():.9f}"
        )

        print()
        print("=" * 100)
        print("BOS MAX-SOURCE FRACTION")
        print("=" * 100)

        for mode in [
            "TT",
            "SYM",
            "T128",
            "128T",
            "128128",
        ]:
            x = df[
                f"bosmax_{mode}"
            ]

            print(
                f"{mode:8s} "
                f"mean={x.mean():.9f}  "
                f"median={x.median():.9f}"
            )

        print()
        print("=" * 100)
        print("REFLECTION / DIRECT RMS")
        print("=" * 100)

        for mode in [
            "TT",
            "SYM",
            "T128",
            "128T",
            "128128",
        ]:
            x = df[
                f"reflection_{mode}"
            ]

            print(
                f"{mode:8s} "
                f"mean={x.mean():.9f}  "
                f"median={x.median():.9f}  "
                f"max={x.max():.9f}"
            )

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
