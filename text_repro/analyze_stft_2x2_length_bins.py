from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


ROOT = Path(
    "outputs/text/stsbenchmark"
)

SEEDS = [0, 1, 2]

BIN_ORDER = [
    "<=16",
    "17-24",
    ">24",
]


# ============================================================
# Helpers
# ============================================================

def sp(x, y):
    return float(
        spearmanr(x, y)[0]
    )


def pr(x, y):
    return float(
        pearsonr(x, y)[0]
    )


def resolve_experiment(
    exact_name,
    glob_pattern,
):
    """
    Prefer exact experiment name.
    If it does not exist, search by a conservative glob.
    """

    exact = (
        ROOT
        / "experiments"
        / exact_name
    )

    if exact.exists():
        return exact

    matches = list(
        (
            ROOT / "experiments"
        ).glob(
            glob_pattern
        )
    )

    if len(matches) == 1:
        print(
            f"Resolved {exact_name} -> "
            f"{matches[0].name}"
        )
        return matches[0]

    raise RuntimeError(
        f"Could not uniquely resolve {exact_name}. "
        f"Matches: {[x.name for x in matches]}"
    )


# ============================================================
# Resolve models
# ============================================================

E2_ROOT = resolve_experiment(
    "E2_stft_flag_w16_h8_rect",
    "*w16*h8*rect*",
)

E3_ROOT = resolve_experiment(
    "E3_stft_flag_w8_h4_rect",
    "*w8*h4*rect*",
)

E11_ROOT = resolve_experiment(
    "E11_stft_flag_w8_h8_rect",
    "*w8*h8*rect*",
)

E12_ROOT = resolve_experiment(
    "E12_stft_flag_w16_h16_rect",
    "*w16*h16*rect*",
)


MODEL_ROOTS = {
    "FLaG":
        ROOT / "FLaG",

    "E2_16_8":
        E2_ROOT,

    "E3_8_4":
        E3_ROOT,

    "E11_8_8":
        E11_ROOT,

    "E12_16_16":
        E12_ROOT,
}


# ============================================================
# Recover pair lengths from E7
# ============================================================

e7_path = (
    ROOT
    / "probes"
    / "E7_token_knockout"
    / "summary"
    / "aligned_seed_token_responses.csv"
)

if not e7_path.exists():
    raise FileNotFoundError(
        f"Missing length source: {e7_path}"
    )

e7 = pd.read_csv(
    e7_path
)

print("E7 columns:")
print(e7.columns.tolist())


# Pick one model only because sentence lengths are identical
# across pooling methods.
if "model" in e7.columns:
    available_models = (
        e7["model"]
        .dropna()
        .unique()
        .tolist()
    )

    if "FLaG" in available_models:
        e7 = e7[
            e7["model"] == "FLaG"
        ].copy()
    else:
        chosen = available_models[0]

        print(
            "FLaG not found in E7 model column; "
            f"using {chosen} for lengths."
        )

        e7 = e7[
            e7["model"] == chosen
        ].copy()


required = {
    "pair_id",
    "side",
    "content_len",
}

missing = (
    required
    - set(e7.columns)
)

if missing:
    raise RuntimeError(
        f"E7 missing columns: {missing}"
    )


# Many token rows exist for each sentence.
# Collapse to one row per pair / side.
sentence_lengths = (
    e7[
        [
            "pair_id",
            "side",
            "content_len",
        ]
    ]
    .drop_duplicates()
)


dup_check = (
    sentence_lengths
    .groupby(
        [
            "pair_id",
            "side",
        ]
    )
    .size()
)

if dup_check.max() != 1:
    raise RuntimeError(
        "Multiple different content lengths found "
        "for the same pair/side."
    )


pair_lengths = (
    sentence_lengths
    .groupby(
        "pair_id"
    )
    .agg(
        content_len1=(
            "content_len",
            lambda x: (
                sentence_lengths.loc[
                    x.index
                ]
                .sort_values("side")
                ["content_len"]
                .iloc[0]
            )
        ),

        content_len2=(
            "content_len",
            lambda x: (
                sentence_lengths.loc[
                    x.index
                ]
                .sort_values("side")
                ["content_len"]
                .iloc[-1]
            )
        ),
    )
    .reset_index()
)


# Simpler / safer reconstruction using pivot
pivot = (
    sentence_lengths
    .pivot(
        index="pair_id",
        columns="side",
        values="content_len",
    )
    .reset_index()
)

if 1 not in pivot.columns or 2 not in pivot.columns:
    raise RuntimeError(
        "Expected sides 1 and 2 in E7."
    )

pair_lengths = pd.DataFrame({
    "pair_id":
        pivot["pair_id"].astype(int),

    "content_len1":
        pivot[1].astype(int),

    "content_len2":
        pivot[2].astype(int),
})


# Restore RoBERTa <s>, </s>.
pair_lengths[
    "pair_max_len"
] = (
    pair_lengths[
        [
            "content_len1",
            "content_len2",
        ]
    ].max(axis=1)
    + 2
)


pair_lengths[
    "length_bin"
] = pd.cut(
    pair_lengths[
        "pair_max_len"
    ],
    bins=[
        -np.inf,
        16,
        24,
        np.inf,
    ],
    labels=BIN_ORDER,
    ordered=True,
)


pair_lengths = (
    pair_lengths
    .sort_values("pair_id")
    .reset_index(drop=True)
)


print()
print("Recovered pair lengths:")
print(
    pair_lengths[
        "length_bin"
    ].value_counts(
        sort=False
    )
)

print(
    "pairs =",
    len(pair_lengths),
)


if len(pair_lengths) != 1379:
    raise RuntimeError(
        f"Expected 1379 pairs, got "
        f"{len(pair_lengths)}"
    )

if not np.array_equal(
    pair_lengths["pair_id"].to_numpy(),
    np.arange(1379),
):
    raise RuntimeError(
        "pair_id is not exactly 0..1378"
    )


# ============================================================
# Load every model prediction
# ============================================================

all_rows = []

reference_true = None

for model, root in MODEL_ROOTS.items():

    print()
    print(
        f"Loading {model}: {root}"
    )

    for seed in SEEDS:

        path = (
            root
            / f"seed_{seed}"
            / "test_predictions.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}"
            )

        pred = pd.read_csv(
            path
        )

        if len(pred) != 1379:
            raise RuntimeError(
                f"{model} seed {seed}: "
                f"expected 1379 rows, "
                f"got {len(pred)}"
            )

        if not {
            "prediction",
            "true",
        }.issubset(
            pred.columns
        ):
            raise RuntimeError(
                f"Unexpected columns in {path}: "
                f"{pred.columns.tolist()}"
            )

        true = (
            pred["true"]
            .to_numpy(
                dtype=float
            )
        )

        if reference_true is None:
            reference_true = true
        else:
            if not np.allclose(
                true,
                reference_true,
                atol=1e-8,
                rtol=0,
            ):
                raise RuntimeError(
                    f"Gold labels mismatch: "
                    f"{model} seed {seed}"
                )

        tmp = pair_lengths.copy()

        tmp["model"] = model
        tmp["seed"] = seed

        tmp["prediction"] = (
            pred["prediction"]
            .to_numpy(
                dtype=float
            )
        )

        tmp["true"] = true

        all_rows.append(
            tmp
        )


df = pd.concat(
    all_rows,
    ignore_index=True,
)


# ============================================================
# Metrics per model / seed / length
# ============================================================

metric_rows = []

for (
    model,
    seed,
    length_bin,
), g in df.groupby(
    [
        "model",
        "seed",
        "length_bin",
    ],
    observed=True,
):

    metric_rows.append({
        "model":
            model,

        "seed":
            int(seed),

        "length_bin":
            str(length_bin),

        "pairs":
            len(g),

        "spearman":
            sp(
                g["prediction"],
                g["true"],
            ),

        "pearson":
            pr(
                g["prediction"],
                g["true"],
            ),
    })


metrics = pd.DataFrame(
    metric_rows
)


# ============================================================
# Summary across seeds
# ============================================================

summary = (
    metrics
    .groupby(
        [
            "model",
            "length_bin",
        ],
        observed=True,
    )
    .agg(
        pairs=(
            "pairs",
            "first",
        ),

        spearman_mean=(
            "spearman",
            "mean",
        ),

        spearman_std=(
            "spearman",
            "std",
        ),

        pearson_mean=(
            "pearson",
            "mean",
        ),

        pearson_std=(
            "pearson",
            "std",
        ),
    )
    .reset_index()
)


# ============================================================
# Paired delta versus FLaG
# ============================================================

flag = (
    metrics[
        metrics["model"]
        == "FLaG"
    ]
    [
        [
            "seed",
            "length_bin",
            "spearman",
            "pearson",
        ]
    ]
    .rename(
        columns={
            "spearman":
                "flag_spearman",

            "pearson":
                "flag_pearson",
        }
    )
)


delta_rows = []

for model in [
    "E2_16_8",
    "E3_8_4",
    "E11_8_8",
    "E12_16_16",
]:

    x = (
        metrics[
            metrics["model"]
            == model
        ]
        .merge(
            flag,
            on=[
                "seed",
                "length_bin",
            ],
            how="inner",
        )
    )

    x[
        "spearman_delta"
    ] = (
        x["spearman"]
        - x["flag_spearman"]
    )

    x[
        "pearson_delta"
    ] = (
        x["pearson"]
        - x["flag_pearson"]
    )

    delta_rows.append(
        x[
            [
                "model",
                "seed",
                "length_bin",
                "spearman_delta",
                "pearson_delta",
            ]
        ]
    )


deltas = pd.concat(
    delta_rows,
    ignore_index=True,
)


delta_summary = (
    deltas
    .groupby(
        [
            "model",
            "length_bin",
        ],
        observed=True,
    )
    .agg(
        spearman_delta=(
            "spearman_delta",
            "mean",
        ),

        spearman_delta_std=(
            "spearman_delta",
            "std",
        ),

        pearson_delta=(
            "pearson_delta",
            "mean",
        ),

        pearson_delta_std=(
            "pearson_delta",
            "std",
        ),
    )
    .reset_index()
)


# ============================================================
# 2x2 factorial-style contrasts
# ============================================================

wide_sp = (
    metrics
    .pivot(
        index=[
            "seed",
            "length_bin",
        ],
        columns="model",
        values="spearman",
    )
    .reset_index()
)


wide_pr = (
    metrics
    .pivot(
        index=[
            "seed",
            "length_bin",
        ],
        columns="model",
        values="pearson",
    )
    .reset_index()
)


contrast_rows = []

for _, r in wide_sp.iterrows():

    seed = int(
        r["seed"]
    )

    length_bin = str(
        r["length_bin"]
    )

    pr_row = wide_pr[
        (wide_pr["seed"] == seed)
        & (
            wide_pr["length_bin"]
            .astype(str)
            == length_bin
        )
    ].iloc[0]

    contrasts = {
        # overlap effect:
        # overlap minus no-overlap
        "overlap_effect_w8":
            (
                r["E3_8_4"]
                - r["E11_8_8"],
                pr_row["E3_8_4"]
                - pr_row["E11_8_8"],
            ),

        "overlap_effect_w16":
            (
                r["E2_16_8"]
                - r["E12_16_16"],
                pr_row["E2_16_8"]
                - pr_row["E12_16_16"],
            ),

        # window-size effect:
        # win8 minus win16
        "window8_vs16_overlap":
            (
                r["E3_8_4"]
                - r["E2_16_8"],
                pr_row["E3_8_4"]
                - pr_row["E2_16_8"],
            ),

        "window8_vs16_no_overlap":
            (
                r["E11_8_8"]
                - r["E12_16_16"],
                pr_row["E11_8_8"]
                - pr_row["E12_16_16"],
            ),
    }

    for name, (
        sp_delta,
        pr_delta,
    ) in contrasts.items():

        contrast_rows.append({
            "seed":
                seed,

            "length_bin":
                length_bin,

            "contrast":
                name,

            "spearman_delta":
                float(sp_delta),

            "pearson_delta":
                float(pr_delta),
        })


contrasts = pd.DataFrame(
    contrast_rows
)


contrast_summary = (
    contrasts
    .groupby(
        [
            "length_bin",
            "contrast",
        ],
        observed=True,
    )
    .agg(
        spearman_delta=(
            "spearman_delta",
            "mean",
        ),

        spearman_std=(
            "spearman_delta",
            "std",
        ),

        pearson_delta=(
            "pearson_delta",
            "mean",
        ),

        pearson_std=(
            "pearson_delta",
            "std",
        ),
    )
    .reset_index()
)


# ============================================================
# Save
# ============================================================

OUT = (
    ROOT
    / "probes"
    / "E12_2x2_length_analysis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


metrics.to_csv(
    OUT
    / "per_seed_metrics_by_length.csv",
    index=False,
)

summary.to_csv(
    OUT
    / "summary_by_length.csv",
    index=False,
)

deltas.to_csv(
    OUT
    / "per_seed_delta_vs_flag.csv",
    index=False,
)

delta_summary.to_csv(
    OUT
    / "delta_vs_flag_summary.csv",
    index=False,
)

contrasts.to_csv(
    OUT
    / "per_seed_2x2_contrasts.csv",
    index=False,
)

contrast_summary.to_csv(
    OUT
    / "2x2_contrast_summary.csv",
    index=False,
)


# ============================================================
# Print useful tables
# ============================================================

print()
print("=" * 100)
print("MEAN PERFORMANCE BY LENGTH")
print("=" * 100)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.6f}"
        ),
    )
)


print()
print("=" * 100)
print("DELTA VS FLaG BY LENGTH")
print("positive = better than FLaG")
print("=" * 100)

print(
    delta_summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.6f}"
        ),
    )
)


print()
print("=" * 100)
print("2 x 2 CONTRASTS BY LENGTH")
print("=" * 100)

print(
    contrast_summary.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.6f}"
        ),
    )
)


print()
print("=" * 100)
print("E12 PER-SEED DELTA VS FLaG")
print("=" * 100)

print(
    deltas[
        deltas["model"]
        == "E12_16_16"
    ].to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.6f}"
        ),
    )
)

print()
print(
    "Saved to:",
    OUT,
)
