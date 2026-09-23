from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


ROOT = Path(
    "outputs/text/stsbenchmark/probes"
)

E6_ROOT = (
    ROOT / "E6_dct_frequency_knockout"
)

E7_ROOT = (
    ROOT / "E7_token_knockout"
)

OUT = (
    ROOT / "length_stratified_analysis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


FLAG = "FLaG"
E3 = "E3_STFT_w8_h4_rect"

BIN_ORDER = [
    "<=16",
    "17-24",
    ">24",
]


def add_length_bin(df, col):
    df = df.copy()

    df["length_bin"] = pd.cut(
        df[col],
        bins=[
            -np.inf,
            16,
            24,
            np.inf,
        ],
        labels=BIN_ORDER,
        ordered=True,
    )

    return df


def safe_spearman(x, y):
    if len(x) < 3:
        return np.nan

    return float(
        spearmanr(x, y)[0]
    )


def safe_pearson(x, y):
    if len(x) < 3:
        return np.nan

    return float(
        pearsonr(x, y)[0]
    )


# =========================================================
# E6: DCT frequency knockout by pair length
# =========================================================

def analyze_e6():
    print()
    print("=" * 80)
    print("E6 LENGTH-STRATIFIED FREQUENCY KNOCKOUT")
    print("=" * 80)

    files = list(
        E6_ROOT.glob(
            "*/seed_*/per_sample_band_responses.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No E6 per-sample files found."
        )

    df = pd.concat(
        [
            pd.read_csv(path)
            for path in files
        ],
        ignore_index=True,
    )

    # E6 saved content-token lengths.
    # +2 restores <s> and </s>, matching our earlier
    # RoBERTa sentence-length definition.
    df["pair_max_len"] = (
        df[
            [
                "content_len1",
                "content_len2",
            ]
        ].max(axis=1)
        + 2
    )

    df = add_length_bin(
        df,
        "pair_max_len",
    )

    # -----------------------------------------------------
    # Paper-style:
    # average each pair's perturbation response
    # across aligned seeds first.
    # -----------------------------------------------------

    aligned = (
        df
        .groupby(
            [
                "model",
                "band",
                "pair_id",
                "pair_max_len",
                "length_bin",
            ],
            observed=True,
        )
        .agg(
            aligned_seed_count=(
                "seed",
                "count",
            ),

            abs_sqerr_change=(
                "abs_sqerr_change",
                "mean",
            ),

            abs_pred_change=(
                "abs_pred_change",
                "mean",
            ),
        )
        .reset_index()
    )

    aligned.to_csv(
        OUT
        / "E6_aligned_pair_responses_by_length.csv",
        index=False,
    )

    response_summary = (
        aligned
        .groupby(
            [
                "model",
                "length_bin",
                "band",
            ],
            observed=True,
        )
        .agg(
            pairs=(
                "pair_id",
                "count",
            ),

            mean_abs_sqerr_change=(
                "abs_sqerr_change",
                "mean",
            ),

            median_abs_sqerr_change=(
                "abs_sqerr_change",
                "median",
            ),

            mean_abs_pred_change=(
                "abs_pred_change",
                "mean",
            ),
        )
        .reset_index()
    )

    response_summary.to_csv(
        OUT
        / "E6_band_response_by_length.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Spearman / Pearson drop inside each length bin,
    # first per seed, then average across seeds.
    # -----------------------------------------------------

    metric_rows = []

    grouped = df.groupby(
        [
            "model",
            "seed",
            "band",
            "length_bin",
        ],
        observed=True,
    )

    for (
        model,
        seed,
        band,
        length_bin,
    ), g in grouped:

        base_sp = safe_spearman(
            g["baseline_pred"],
            g["gold"],
        )

        knock_sp = safe_spearman(
            g["knockout_pred"],
            g["gold"],
        )

        base_pr = safe_pearson(
            g["baseline_pred"],
            g["gold"],
        )

        knock_pr = safe_pearson(
            g["knockout_pred"],
            g["gold"],
        )

        metric_rows.append({
            "model":
                model,

            "seed":
                seed,

            "band":
                band,

            "length_bin":
                length_bin,

            "pairs":
                len(g),

            "baseline_spearman":
                base_sp,

            "knockout_spearman":
                knock_sp,

            "spearman_drop":
                base_sp
                - knock_sp,

            "baseline_pearson":
                base_pr,

            "knockout_pearson":
                knock_pr,

            "pearson_drop":
                base_pr
                - knock_pr,
        })

    per_seed_metrics = pd.DataFrame(
        metric_rows
    )

    per_seed_metrics.to_csv(
        OUT
        / "E6_per_seed_metric_drop_by_length.csv",
        index=False,
    )

    metric_summary = (
        per_seed_metrics
        .groupby(
            [
                "model",
                "length_bin",
                "band",
            ],
            observed=True,
        )
        .agg(
            seeds=(
                "seed",
                "count",
            ),

            pairs_per_seed=(
                "pairs",
                "first",
            ),

            mean_spearman_drop=(
                "spearman_drop",
                "mean",
            ),

            std_spearman_drop=(
                "spearman_drop",
                "std",
            ),

            mean_pearson_drop=(
                "pearson_drop",
                "mean",
            ),
        )
        .reset_index()
    )

    metric_summary.to_csv(
        OUT
        / "E6_metric_drop_by_length.csv",
        index=False,
    )

    # =====================================================
    # B0 direct comparison
    # =====================================================

    b0_response = response_summary[
        response_summary["band"] == 0
    ]

    flag_resp = (
        b0_response[
            b0_response["model"] == FLAG
        ]
        [
            [
                "length_bin",
                "pairs",
                "mean_abs_sqerr_change",
                "mean_abs_pred_change",
            ]
        ]
        .rename(
            columns={
                "pairs":
                    "pairs",

                "mean_abs_sqerr_change":
                    "flag_abs_sqerr",

                "mean_abs_pred_change":
                    "flag_abs_pred",
            }
        )
    )

    e3_resp = (
        b0_response[
            b0_response["model"] == E3
        ]
        [
            [
                "length_bin",
                "mean_abs_sqerr_change",
                "mean_abs_pred_change",
            ]
        ]
        .rename(
            columns={
                "mean_abs_sqerr_change":
                    "e3_abs_sqerr",

                "mean_abs_pred_change":
                    "e3_abs_pred",
            }
        )
    )

    b0_metric = metric_summary[
        metric_summary["band"] == 0
    ]

    flag_metric = (
        b0_metric[
            b0_metric["model"] == FLAG
        ]
        [
            [
                "length_bin",
                "mean_spearman_drop",
                "std_spearman_drop",
            ]
        ]
        .rename(
            columns={
                "mean_spearman_drop":
                    "flag_spearman_drop",

                "std_spearman_drop":
                    "flag_spearman_drop_std",
            }
        )
    )

    e3_metric = (
        b0_metric[
            b0_metric["model"] == E3
        ]
        [
            [
                "length_bin",
                "mean_spearman_drop",
                "std_spearman_drop",
            ]
        ]
        .rename(
            columns={
                "mean_spearman_drop":
                    "e3_spearman_drop",

                "std_spearman_drop":
                    "e3_spearman_drop_std",
            }
        )
    )

    b0_compare = (
        flag_resp
        .merge(
            e3_resp,
            on="length_bin",
        )
        .merge(
            flag_metric,
            on="length_bin",
        )
        .merge(
            e3_metric,
            on="length_bin",
        )
    )

    b0_compare[
        "e3_minus_flag_abs_sqerr"
    ] = (
        b0_compare["e3_abs_sqerr"]
        - b0_compare["flag_abs_sqerr"]
    )

    b0_compare[
        "e3_minus_flag_abs_pred"
    ] = (
        b0_compare["e3_abs_pred"]
        - b0_compare["flag_abs_pred"]
    )

    b0_compare[
        "e3_minus_flag_spearman_drop"
    ] = (
        b0_compare["e3_spearman_drop"]
        - b0_compare["flag_spearman_drop"]
    )

    b0_compare.to_csv(
        OUT
        / "E6_B0_length_comparison.csv",
        index=False,
    )

    print()
    print("----------------------------------------")
    print("E6 B0 / DC SENSITIVITY BY LENGTH")
    print("positive E3-FLaG = E3 more sensitive")
    print("----------------------------------------")

    print(
        b0_compare.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    # =====================================================
    # All frequency bands for SHORT pairs
    # =====================================================

    short_resp = response_summary[
        response_summary["length_bin"]
        == "<=16"
    ]

    short_metric = metric_summary[
        metric_summary["length_bin"]
        == "<=16"
    ]

    flag_sr = (
        short_resp[
            short_resp["model"] == FLAG
        ]
        [
            [
                "band",
                "pairs",
                "mean_abs_sqerr_change",
                "mean_abs_pred_change",
            ]
        ]
        .rename(
            columns={
                "mean_abs_sqerr_change":
                    "flag_abs_sqerr",

                "mean_abs_pred_change":
                    "flag_abs_pred",
            }
        )
    )

    e3_sr = (
        short_resp[
            short_resp["model"] == E3
        ]
        [
            [
                "band",
                "mean_abs_sqerr_change",
                "mean_abs_pred_change",
            ]
        ]
        .rename(
            columns={
                "mean_abs_sqerr_change":
                    "e3_abs_sqerr",

                "mean_abs_pred_change":
                    "e3_abs_pred",
            }
        )
    )

    flag_sm = (
        short_metric[
            short_metric["model"] == FLAG
        ]
        [
            [
                "band",
                "mean_spearman_drop",
            ]
        ]
        .rename(
            columns={
                "mean_spearman_drop":
                    "flag_spearman_drop",
            }
        )
    )

    e3_sm = (
        short_metric[
            short_metric["model"] == E3
        ]
        [
            [
                "band",
                "mean_spearman_drop",
            ]
        ]
        .rename(
            columns={
                "mean_spearman_drop":
                    "e3_spearman_drop",
            }
        )
    )

    short_compare = (
        flag_sr
        .merge(
            e3_sr,
            on="band",
        )
        .merge(
            flag_sm,
            on="band",
        )
        .merge(
            e3_sm,
            on="band",
        )
    )

    short_compare[
        "e3_minus_flag_abs_sqerr"
    ] = (
        short_compare["e3_abs_sqerr"]
        - short_compare["flag_abs_sqerr"]
    )

    short_compare[
        "e3_minus_flag_spearman_drop"
    ] = (
        short_compare["e3_spearman_drop"]
        - short_compare["flag_spearman_drop"]
    )

    short_compare.to_csv(
        OUT
        / "E6_short_pair_all_band_comparison.csv",
        index=False,
    )

    print()
    print("----------------------------------------")
    print("E6 SHORT PAIRS (<=16): ALL BANDS")
    print("----------------------------------------")

    print(
        short_compare.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )


# =========================================================
# E7: token knockout by pair length
# =========================================================

def analyze_e7():
    print()
    print("=" * 80)
    print("E7 LENGTH-STRATIFIED TOKEN KNOCKOUT")
    print("=" * 80)

    path = (
        E7_ROOT
        / "summary"
        / "aligned_seed_token_responses.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}"
        )

    df = pd.read_csv(
        path
    )

    # Pair length = longer sentence length.
    # content_len excludes <s>, </s>.
    pair_lengths = (
        df
        .groupby(
            [
                "model",
                "pair_id",
            ]
        )["content_len"]
        .max()
        .add(2)
        .rename("pair_max_len")
        .reset_index()
    )

    df = df.merge(
        pair_lengths,
        on=[
            "model",
            "pair_id",
        ],
        how="left",
    )

    df = add_length_bin(
        df,
        "pair_max_len",
    )

    # -----------------------------------------------------
    # Reproduce paper-style sentence-level heterogeneity,
    # now stratified by pair length.
    # -----------------------------------------------------

    def population_std(x):
        return float(
            np.std(
                x.to_numpy(
                    dtype=np.float64
                ),
                ddof=0,
            )
        )

    sentence = (
        df
        .groupby(
            [
                "model",
                "pair_id",
                "side",
                "length_bin",
            ],
            observed=True,
        )
        .agg(
            content_len=(
                "content_len",
                "first",
            ),

            positional_response_std=(
                "abs_sqerr_change",
                population_std,
            ),

            mean_token_response=(
                "abs_sqerr_change",
                "mean",
            ),

            max_token_response=(
                "abs_sqerr_change",
                "max",
            ),

            mean_abs_pred_change=(
                "abs_pred_change",
                "mean",
            ),
        )
        .reset_index()
    )

    sentence.to_csv(
        OUT
        / "E7_sentence_heterogeneity_by_length_raw.csv",
        index=False,
    )

    summary = (
        sentence
        .groupby(
            [
                "model",
                "length_bin",
            ],
            observed=True,
        )
        .agg(
            sentences=(
                "pair_id",
                "count",
            ),

            mean_positional_std=(
                "positional_response_std",
                "mean",
            ),

            median_positional_std=(
                "positional_response_std",
                "median",
            ),

            mean_token_response=(
                "mean_token_response",
                "mean",
            ),

            mean_max_token_response=(
                "max_token_response",
                "mean",
            ),

            mean_abs_pred_change=(
                "mean_abs_pred_change",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUT
        / "E7_heterogeneity_by_length.csv",
        index=False,
    )

    print()
    print("----------------------------------------")
    print("E7 POSITIONAL HETEROGENEITY BY LENGTH")
    print("----------------------------------------")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    # =====================================================
    # Direct E3 - FLaG by length
    # =====================================================

    flag = (
        summary[
            summary["model"] == FLAG
        ]
        .drop(
            columns="model"
        )
        .rename(
            columns={
                "sentences":
                    "sentences",

                "mean_positional_std":
                    "flag_positional_std",

                "median_positional_std":
                    "flag_median_positional_std",

                "mean_token_response":
                    "flag_token_response",

                "mean_max_token_response":
                    "flag_max_token_response",

                "mean_abs_pred_change":
                    "flag_abs_pred",
            }
        )
    )

    e3 = (
        summary[
            summary["model"] == E3
        ]
        .drop(
            columns=[
                "model",
                "sentences",
            ]
        )
        .rename(
            columns={
                "mean_positional_std":
                    "e3_positional_std",

                "median_positional_std":
                    "e3_median_positional_std",

                "mean_token_response":
                    "e3_token_response",

                "mean_max_token_response":
                    "e3_max_token_response",

                "mean_abs_pred_change":
                    "e3_abs_pred",
            }
        )
    )

    compare = flag.merge(
        e3,
        on="length_bin",
    )

    compare[
        "e3_minus_flag_positional_std"
    ] = (
        compare["e3_positional_std"]
        - compare["flag_positional_std"]
    )

    compare[
        "e3_minus_flag_token_response"
    ] = (
        compare["e3_token_response"]
        - compare["flag_token_response"]
    )

    compare[
        "e3_minus_flag_abs_pred"
    ] = (
        compare["e3_abs_pred"]
        - compare["flag_abs_pred"]
    )

    compare.to_csv(
        OUT
        / "E7_flag_vs_e3_by_length.csv",
        index=False,
    )

    print()
    print("----------------------------------------")
    print("E7 FLaG VS E3 BY LENGTH")
    print("positive difference = E3 more sensitive")
    print("----------------------------------------")

    print(
        compare.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    # =====================================================
    # Position profile within each length group
    # =====================================================

    pos = (
        df
        .groupby(
            [
                "model",
                "length_bin",
                "position_bin",
            ],
            observed=True,
        )
        .agg(
            tokens=(
                "content_pos",
                "count",
            ),

            mean_abs_sqerr_change=(
                "abs_sqerr_change",
                "mean",
            ),

            mean_abs_pred_change=(
                "abs_pred_change",
                "mean",
            ),
        )
        .reset_index()
    )

    pos["position_region"] = (
        pos["position_bin"].map({
            0: "start",
            1: "early",
            2: "middle",
            3: "late",
            4: "end",
        })
    )

    pos.to_csv(
        OUT
        / "E7_position_profile_by_length.csv",
        index=False,
    )

    # Short-pair position difference
    short = pos[
        pos["length_bin"] == "<=16"
    ]

    flag_short = (
        short[
            short["model"] == FLAG
        ]
        [
            [
                "position_bin",
                "position_region",
                "tokens",
                "mean_abs_sqerr_change",
                "mean_abs_pred_change",
            ]
        ]
        .rename(
            columns={
                "mean_abs_sqerr_change":
                    "flag_abs_sqerr",

                "mean_abs_pred_change":
                    "flag_abs_pred",
            }
        )
    )

    e3_short = (
        short[
            short["model"] == E3
        ]
        [
            [
                "position_bin",
                "mean_abs_sqerr_change",
                "mean_abs_pred_change",
            ]
        ]
        .rename(
            columns={
                "mean_abs_sqerr_change":
                    "e3_abs_sqerr",

                "mean_abs_pred_change":
                    "e3_abs_pred",
            }
        )
    )

    short_pos_compare = (
        flag_short.merge(
            e3_short,
            on="position_bin",
        )
    )

    short_pos_compare[
        "e3_minus_flag_abs_sqerr"
    ] = (
        short_pos_compare["e3_abs_sqerr"]
        - short_pos_compare["flag_abs_sqerr"]
    )

    short_pos_compare[
        "e3_minus_flag_abs_pred"
    ] = (
        short_pos_compare["e3_abs_pred"]
        - short_pos_compare["flag_abs_pred"]
    )

    short_pos_compare.to_csv(
        OUT
        / "E7_short_pair_position_comparison.csv",
        index=False,
    )

    print()
    print("----------------------------------------")
    print("E7 SHORT PAIRS (<=16): POSITION PROFILE")
    print("----------------------------------------")

    print(
        short_pos_compare.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )


def main():
    analyze_e6()
    analyze_e7()

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(
        "Saved to:",
        OUT,
    )


if __name__ == "__main__":
    main()
