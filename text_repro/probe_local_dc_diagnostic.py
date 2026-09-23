import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from text_repro.probe_token_knockout import (
    build_model,
    build_test_loader,
    pool_hidden,
    correlation_metrics,
)

from text_repro.train_sts import (
    seed_everything,
)


EPS = 1e-12


# =========================================================
# Extract diagnostics from one pooling forward
# =========================================================

@torch.no_grad()
def extract_pool_diagnostics(
    model,
    hidden,
    attention_mask,
    model_key,
):
    """
    Run the normal pooling forward, then inspect the cached
    spectral tokens / latent-attention weights / gate.

    Returns:
        pooled embedding [B,D]
        list[dict], one row per sentence
    """

    pooled = pool_hidden(
        model,
        hidden,
        attention_mask,
    )

    pool = model.pool

    required = [
        "_last_latent_attn_weights",
        "_last_freq_tokens",
        "_last_raw_gate",
    ]

    for name in required:
        if not hasattr(pool, name):
            raise RuntimeError(
                f"Pooling object does not expose {name}"
            )

    # -----------------------------------------------------
    # Attention
    # -----------------------------------------------------

    attn = (
        pool
        ._last_latent_attn_weights
        .detach()
    )

    # Defensive handling:
    # [B,H,L,S] -> mean heads -> [B,L,S]
    if attn.dim() == 4:
        attn = attn.mean(dim=1)

    if attn.dim() != 3:
        raise RuntimeError(
            "Expected attention weights [B,L,S], "
            f"got {tuple(attn.shape)}"
        )

    # Average over latent queries.
    # [B,L,S] -> [B,S]
    token_attn = attn.mean(dim=1)

    # -----------------------------------------------------
    # Raw spectral tokens
    # -----------------------------------------------------

    freq_tokens = (
        pool
        ._last_freq_tokens
        .detach()
    )

    if freq_tokens.dim() != 3:
        raise RuntimeError(
            "Expected freq_tokens [B,S,2D], "
            f"got {tuple(freq_tokens.shape)}"
        )

    B, S, _ = freq_tokens.shape

    if token_attn.shape != (
        B,
        S,
    ):
        raise RuntimeError(
            "Attention/source-token shape mismatch: "
            f"{tuple(token_attn.shape)} vs "
            f"{tuple(freq_tokens.shape)}"
        )

    # Complex spectral energy:
    # concat(real, imag), so sum of squares over the
    # last dimension corresponds to total complex energy.
    spectral_energy = (
        freq_tokens
        .float()
        .pow(2)
        .sum(dim=-1)
    )

    # -----------------------------------------------------
    # Identify valid spectral tokens + DC tokens
    # -----------------------------------------------------

    indices = torch.arange(
        S,
        device=freq_tokens.device,
    ).unsqueeze(0).expand(
        B,
        -1,
    )

    if model_key == "e3":
        if not hasattr(
            pool,
            "_last_stft_freq_mask",
        ):
            raise RuntimeError(
                "E3 pool is missing "
                "_last_stft_freq_mask"
            )

        valid = (
            pool
            ._last_stft_freq_mask
            .detach()
            .bool()
        )

        num_freqs = (
            int(pool.win_length)
            // 2
            + 1
        )

        # Flatten order:
        # frame0: k0,k1,...,kF-1
        # frame1: k0,k1,...,kF-1
        dc_mask = (
            valid
            & (
                indices
                % num_freqs
                == 0
            )
        )

    elif model_key == "flag":
        # Global FLaG has one global frequency sequence.
        valid = torch.ones(
            B,
            S,
            dtype=torch.bool,
            device=freq_tokens.device,
        )

        dc_mask = torch.zeros_like(
            valid
        )

        dc_mask[:, 0] = True

        num_freqs = S

    else:
        raise ValueError(
            f"Unknown model_key={model_key}"
        )

    non_dc_mask = (
        valid
        & ~dc_mask
    )

    n_dc = (
        dc_mask
        .sum(dim=1)
        .float()
    )

    n_non_dc = (
        non_dc_mask
        .sum(dim=1)
        .float()
    )

    n_valid = (
        valid
        .sum(dim=1)
        .float()
    )

    if torch.any(n_dc <= 0):
        raise RuntimeError(
            "A sample has no DC spectral token."
        )

    if torch.any(n_non_dc <= 0):
        raise RuntimeError(
            "A sample has no non-DC spectral token."
        )

    # -----------------------------------------------------
    # Attention statistics
    # -----------------------------------------------------

    dc_attn_mass = (
        token_attn
        * dc_mask.to(
            token_attn.dtype
        )
    ).sum(dim=1)

    non_dc_attn_mass = (
        token_attn
        * non_dc_mask.to(
            token_attn.dtype
        )
    ).sum(dim=1)

    dc_attn_per_token = (
        dc_attn_mass
        / n_dc
    )

    non_dc_attn_per_token = (
        non_dc_attn_mass
        / n_non_dc
    )

    attn_preference_ratio = (
        dc_attn_per_token
        / (
            non_dc_attn_per_token
            + EPS
        )
    )

    # Under uniform attention, expected DC mass is
    # n_dc / n_valid.
    expected_dc_mass = (
        n_dc
        / n_valid
    )

    dc_mass_lift = (
        dc_attn_mass
        / (
            expected_dc_mass
            + EPS
        )
    )

    # Normalized attention entropy.
    p = (
        token_attn
        * valid.to(
            token_attn.dtype
        )
    )

    p = p / (
        p.sum(dim=1, keepdim=True)
        + EPS
    )

    entropy = -(
        p
        * torch.log(
            p + EPS
        )
    ).sum(dim=1)

    normalized_entropy = (
        entropy
        / torch.log(
            n_valid.clamp(min=2.0)
        )
    )

    # -----------------------------------------------------
    # Spectral-energy statistics
    # -----------------------------------------------------

    dc_energy_per_token = (
        spectral_energy
        * dc_mask.to(
            spectral_energy.dtype
        )
    ).sum(dim=1) / n_dc

    non_dc_energy_per_token = (
        spectral_energy
        * non_dc_mask.to(
            spectral_energy.dtype
        )
    ).sum(dim=1) / n_non_dc

    energy_ratio = (
        dc_energy_per_token
        / (
            non_dc_energy_per_token
            + EPS
        )
    )

    # -----------------------------------------------------
    # Gate / latent-summary diagnostics
    # -----------------------------------------------------

    gate = (
        pool
        ._last_raw_gate
        .detach()
        .float()
    )

    gate_mean = gate.mean(dim=-1)
    gate_std = gate.std(
        dim=-1,
        unbiased=False,
    )

    gate_rms = torch.sqrt(
        gate.pow(2).mean(dim=-1)
    )

    if hasattr(
        pool,
        "_last_latent_summary",
    ):
        latent_summary = (
            pool
            ._last_latent_summary
            .detach()
            .float()
        )

        latent_summary_rms = (
            torch.sqrt(
                latent_summary
                .pow(2)
                .mean(dim=-1)
            )
        )
    else:
        latent_summary_rms = (
            torch.full(
                (B,),
                float("nan"),
                device=hidden.device,
            )
        )

    valid_lengths = (
        attention_mask
        .long()
        .sum(dim=1)
    )

    rows = []

    for i in range(B):
        rows.append({
            "valid_len":
                int(
                    valid_lengths[
                        i
                    ].item()
                ),

            "content_len":
                max(
                    0,
                    int(
                        valid_lengths[
                            i
                        ].item()
                    )
                    - 2,
                ),

            "num_spectral_tokens":
                int(
                    n_valid[
                        i
                    ].item()
                ),

            "num_dc_tokens":
                int(
                    n_dc[
                        i
                    ].item()
                ),

            "num_non_dc_tokens":
                int(
                    n_non_dc[
                        i
                    ].item()
                ),

            "num_freqs_per_frame":
                int(num_freqs),

            "dc_attn_mass":
                float(
                    dc_attn_mass[
                        i
                    ].item()
                ),

            "non_dc_attn_mass":
                float(
                    non_dc_attn_mass[
                        i
                    ].item()
                ),

            "dc_attn_per_token":
                float(
                    dc_attn_per_token[
                        i
                    ].item()
                ),

            "non_dc_attn_per_token":
                float(
                    non_dc_attn_per_token[
                        i
                    ].item()
                ),

            "attn_preference_ratio":
                float(
                    attn_preference_ratio[
                        i
                    ].item()
                ),

            "dc_mass_lift":
                float(
                    dc_mass_lift[
                        i
                    ].item()
                ),

            "attention_entropy":
                float(
                    normalized_entropy[
                        i
                    ].item()
                ),

            "dc_energy_per_token":
                float(
                    dc_energy_per_token[
                        i
                    ].item()
                ),

            "non_dc_energy_per_token":
                float(
                    non_dc_energy_per_token[
                        i
                    ].item()
                ),

            "energy_ratio":
                float(
                    energy_ratio[
                        i
                    ].item()
                ),

            "gate_mean":
                float(
                    gate_mean[
                        i
                    ].item()
                ),

            "gate_std":
                float(
                    gate_std[
                        i
                    ].item()
                ),

            "gate_rms":
                float(
                    gate_rms[
                        i
                    ].item()
                ),

            "latent_summary_rms":
                float(
                    latent_summary_rms[
                        i
                    ].item()
                ),
        })

    return pooled, rows


# =========================================================
# One model / seed
# =========================================================

@torch.no_grad()
def run_one(
    model_key,
    model_name,
    seed,
    run_dir,
    output_dir,
    device,
):
    seed_dir = (
        output_dir
        / model_name
        / f"seed_{seed}"
    )

    seed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = (
        seed_dir / "SUCCESS"
    )

    if success.exists():
        print(
            f"{model_name} seed={seed} "
            "already SUCCESS, skip."
        )
        return

    print()
    print("=" * 72)
    print(
        model_name,
        "| seed",
        seed,
    )
    print("=" * 72)

    seed_everything(seed)

    model, config = build_model(
        run_dir,
        device,
    )

    loader, _ = build_test_loader(
        config
    )

    with open(
        run_dir / "metrics.json"
    ) as f:
        stored = json.load(f)

    all_rows = []
    predictions = []
    gold = []

    pair_base = 0

    for batch_idx, (
        tok1,
        tok2,
        score,
    ) in enumerate(
        loader,
        start=1,
    ):
        tok1 = {
            k: v.to(device)
            for k, v in tok1.items()
        }

        tok2 = {
            k: v.to(device)
            for k, v in tok2.items()
        }

        target = (
            score.to(device)
            / 5.0
        )

        B = target.size(0)

        # Backbone once per sentence.
        h1 = model.backbone(
            **tok1
        ).last_hidden_state

        h2 = model.backbone(
            **tok2
        ).last_hidden_state

        # ---------------------------------------------
        # Sentence 1
        # ---------------------------------------------

        z1, rows1 = (
            extract_pool_diagnostics(
                model=model,
                hidden=h1,
                attention_mask=(
                    tok1[
                        "attention_mask"
                    ]
                ),
                model_key=model_key,
            )
        )

        # ---------------------------------------------
        # Sentence 2
        # ---------------------------------------------

        z2, rows2 = (
            extract_pool_diagnostics(
                model=model,
                hidden=h2,
                attention_mask=(
                    tok2[
                        "attention_mask"
                    ]
                ),
                model_key=model_key,
            )
        )

        pred = F.cosine_similarity(
            z1,
            z2,
            dim=-1,
        )

        predictions.extend(
            pred.cpu().tolist()
        )

        gold.extend(
            target.cpu().tolist()
        )

        len1 = (
            tok1["attention_mask"]
            .sum(dim=1)
            .long()
        )

        len2 = (
            tok2["attention_mask"]
            .sum(dim=1)
            .long()
        )

        pair_max_len = torch.maximum(
            len1,
            len2,
        )

        for j in range(B):
            pair_id = (
                pair_base + j
            )

            for side, row in [
                (1, rows1[j]),
                (2, rows2[j]),
            ]:
                row.update({
                    "model":
                        model_name,

                    "seed":
                        seed,

                    "pair_id":
                        pair_id,

                    "side":
                        side,

                    "pair_max_len":
                        int(
                            pair_max_len[
                                j
                            ].item()
                        ),
                })

                all_rows.append(
                    row
                )

        pair_base += B

        if (
            batch_idx % 100 == 0
            or batch_idx == len(loader)
        ):
            print(
                f"batch "
                f"{batch_idx}/{len(loader)}"
            )

    # -----------------------------------------------------
    # Sanity: normal predictions must still match.
    # -----------------------------------------------------

    sp, pr = correlation_metrics(
        predictions,
        gold,
    )

    stored_sp = float(
        stored["test_spearman"]
    )

    stored_pr = float(
        stored["test_pearson"]
    )

    print()
    print("Baseline sanity:")
    print(
        f"computed Spearman = {sp:.6f}"
    )
    print(
        f"stored   Spearman = {stored_sp:.6f}"
    )
    print(
        f"computed Pearson  = {pr:.6f}"
    )
    print(
        f"stored   Pearson  = {stored_pr:.6f}"
    )

    if (
        abs(sp - stored_sp) > 1e-5
        or abs(pr - stored_pr) > 1e-5
    ):
        raise RuntimeError(
            "Baseline reproduction failed."
        )

    print(
        "BASELINE REPRODUCTION PASSED"
    )

    df = pd.DataFrame(
        all_rows
    )

    df["length_bin"] = pd.cut(
        df["pair_max_len"],
        bins=[
            -np.inf,
            16,
            24,
            np.inf,
        ],
        labels=[
            "<=16",
            "17-24",
            ">24",
        ],
        ordered=True,
    )

    df.to_csv(
        seed_dir
        / "per_sentence_dc_diagnostics.csv",
        index=False,
    )

    overall = {
        "model":
            model_name,

        "seed":
            seed,

        "sentences":
            len(df),

        "mean_attn_preference_ratio":
            float(
                df[
                    "attn_preference_ratio"
                ].mean()
            ),

        "median_attn_preference_ratio":
            float(
                df[
                    "attn_preference_ratio"
                ].median()
            ),

        "mean_dc_mass_lift":
            float(
                df[
                    "dc_mass_lift"
                ].mean()
            ),

        "mean_energy_ratio":
            float(
                df[
                    "energy_ratio"
                ].mean()
            ),

        "median_energy_ratio":
            float(
                df[
                    "energy_ratio"
                ].median()
            ),

        "mean_gate_mean":
            float(
                df["gate_mean"].mean()
            ),

        "mean_gate_std":
            float(
                df["gate_std"].mean()
            ),

        "mean_attention_entropy":
            float(
                df[
                    "attention_entropy"
                ].mean()
            ),
    }

    with open(
        seed_dir / "summary.json",
        "w",
    ) as f:
        json.dump(
            overall,
            f,
            indent=2,
        )

    success.touch()

    print()
    print(
        pd.DataFrame(
            [overall]
        ).to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    del model
    torch.cuda.empty_cache()


# =========================================================
# Aggregate existing completed runs
# =========================================================

def aggregate(output_dir):
    files = list(
        output_dir.glob(
            "*/seed_*/per_sentence_dc_diagnostics.csv"
        )
    )

    if not files:
        return

    df = pd.concat(
        [
            pd.read_csv(path)
            for path in files
        ],
        ignore_index=True,
    )

    summary_dir = (
        output_dir / "summary"
    )

    summary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Model / seed / pair-length summaries
    # -----------------------------------------------------

    by_seed = (
        df
        .groupby(
            [
                "model",
                "seed",
                "length_bin",
            ],
            observed=True,
        )
        .agg(
            sentences=(
                "pair_id",
                "count",
            ),

            attn_preference_ratio=(
                "attn_preference_ratio",
                "mean",
            ),

            median_attn_preference_ratio=(
                "attn_preference_ratio",
                "median",
            ),

            dc_mass_lift=(
                "dc_mass_lift",
                "mean",
            ),

            energy_ratio=(
                "energy_ratio",
                "mean",
            ),

            attention_entropy=(
                "attention_entropy",
                "mean",
            ),

            gate_mean=(
                "gate_mean",
                "mean",
            ),

            gate_std=(
                "gate_std",
                "mean",
            ),

            latent_summary_rms=(
                "latent_summary_rms",
                "mean",
            ),
        )
        .reset_index()
    )

    by_seed.to_csv(
        summary_dir
        / "dc_diagnostic_by_seed_length.csv",
        index=False,
    )

    overall_seed = (
        df
        .groupby(
            [
                "model",
                "seed",
            ]
        )
        .agg(
            sentences=(
                "pair_id",
                "count",
            ),

            attn_preference_ratio=(
                "attn_preference_ratio",
                "mean",
            ),

            dc_mass_lift=(
                "dc_mass_lift",
                "mean",
            ),

            energy_ratio=(
                "energy_ratio",
                "mean",
            ),

            attention_entropy=(
                "attention_entropy",
                "mean",
            ),

            gate_mean=(
                "gate_mean",
                "mean",
            ),

            gate_std=(
                "gate_std",
                "mean",
            ),
        )
        .reset_index()
    )

    overall_seed.to_csv(
        summary_dir
        / "dc_diagnostic_by_seed.csv",
        index=False,
    )

    print()
    print("=" * 88)
    print("DC DIAGNOSTIC BY SEED / LENGTH")
    print("=" * 88)

    show_cols = [
        "model",
        "seed",
        "length_bin",
        "sentences",
        "attn_preference_ratio",
        "dc_mass_lift",
        "energy_ratio",
        "attention_entropy",
        "gate_mean",
        "gate_std",
    ]

    print(
        by_seed[
            show_cols
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    # -----------------------------------------------------
    # Direct FLaG vs E3 comparison
    # -----------------------------------------------------

    FLAG = "FLaG"
    E3 = "E3_STFT_w8_h4_rect"

    flag = (
        by_seed[
            by_seed["model"] == FLAG
        ]
        [
            [
                "seed",
                "length_bin",
                "attn_preference_ratio",
                "energy_ratio",
                "gate_mean",
                "gate_std",
            ]
        ]
        .rename(
            columns={
                "attn_preference_ratio":
                    "flag_attn_ratio",

                "energy_ratio":
                    "flag_energy_ratio",

                "gate_mean":
                    "flag_gate_mean",

                "gate_std":
                    "flag_gate_std",
            }
        )
    )

    e3 = (
        by_seed[
            by_seed["model"] == E3
        ]
        [
            [
                "seed",
                "length_bin",
                "attn_preference_ratio",
                "energy_ratio",
                "gate_mean",
                "gate_std",
            ]
        ]
        .rename(
            columns={
                "attn_preference_ratio":
                    "e3_attn_ratio",

                "energy_ratio":
                    "e3_energy_ratio",

                "gate_mean":
                    "e3_gate_mean",

                "gate_std":
                    "e3_gate_std",
            }
        )
    )

    compare = flag.merge(
        e3,
        on=[
            "seed",
            "length_bin",
        ],
        how="inner",
    )

    if len(compare) > 0:
        compare[
            "e3_minus_flag_attn_ratio"
        ] = (
            compare["e3_attn_ratio"]
            - compare["flag_attn_ratio"]
        )

        compare[
            "e3_minus_flag_energy_ratio"
        ] = (
            compare["e3_energy_ratio"]
            - compare["flag_energy_ratio"]
        )

        # ---------------------------------------------
        # Attach E6 B0 relative-gap change.
        # ---------------------------------------------

        e6_path = Path(
            "outputs/text/stsbenchmark/"
            "probes/E6_dct_frequency_knockout/"
            "summary/E6_relative_gap_per_seed.csv"
        )

        if e6_path.exists():
            e6 = pd.read_csv(
                e6_path
            )

            e6 = e6[
                e6["band"] == 0
            ][
                [
                    "seed",
                    "length_bin",
                    "sp_gap_change",
                    "pr_gap_change",
                ]
            ]

            compare = compare.merge(
                e6,
                on=[
                    "seed",
                    "length_bin",
                ],
                how="left",
            )

        compare.to_csv(
            summary_dir
            / "dc_internal_vs_B0_knockout.csv",
            index=False,
        )

        print()
        print("=" * 88)
        print(
            "INTERNAL DC BEHAVIOR VS B0 KNOCKOUT"
        )
        print(
            "negative sp_gap_change = "
            "B0 knockout hurts E3 relatively more"
        )
        print("=" * 88)

        print(
            compare.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.6f}"
                ),
            )
        )

    print()
    print(
        "Saved to:",
        summary_dir,
    )


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "flag",
            "e3",
        ],
        default=[
            "flag",
            "e3",
        ],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[
            0,
            1,
            2,
        ],
    )

    parser.add_argument(
        "--root",
        default=(
            "/home/data/home/wwr_lumos/"
            "AMPCliff/outputs/text/stsbenchmark"
        ),
    )

    args = parser.parse_args()

    root = Path(
        args.root
    )

    specs = {
        "flag": {
            "name":
                "FLaG",

            "run_root":
                root / "FLaG",
        },

        "e3": {
            "name":
                "E3_STFT_w8_h4_rect",

            "run_root":
                root
                / "experiments"
                / "E3_stft_flag_w8_h4_rect",
        },
    }

    output_dir = (
        root
        / "probes"
        / "E8_local_dc_diagnostic"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "device:",
        device,
    )

    for model_key in args.models:
        spec = specs[
            model_key
        ]

        for seed in args.seeds:
            run_one(
                model_key=model_key,

                model_name=(
                    spec["name"]
                ),

                seed=seed,

                run_dir=(
                    spec["run_root"]
                    / f"seed_{seed}"
                ),

                output_dir=output_dir,

                device=device,
            )

    aggregate(
        output_dir
    )


if __name__ == "__main__":
    main()
