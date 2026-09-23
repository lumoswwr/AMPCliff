import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_from_disk
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

from text_repro.train_sts import (
    STSDataset,
    STSCollator,
    SentenceEncoder,
    seed_everything,
)


# =========================================================
# Basic utilities
# =========================================================

def correlation_metrics(pred, gold):
    pred = np.asarray(pred, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.float64)

    return (
        float(spearmanr(pred, gold)[0]),
        float(pearsonr(pred, gold)[0]),
    )


def build_model(run_dir, device):
    with open(run_dir / "config.json") as f:
        config = json.load(f)

    candidate_kwargs = {
        "model_path":
            config["model_path"],

        "pooling":
            config["pooling"],

        "stft_win_length":
            config.get(
                "stft_win_length",
                16,
            ),

        "stft_hop_length":
            config.get(
                "stft_hop_length",
                8,
            ),

        "stft_window_type":
            config.get(
                "stft_window_type",
                "rect",
            ),

        "stft_center":
            config.get(
                "stft_center",
                False,
            ),
    }

    signature = inspect.signature(
        SentenceEncoder.__init__
    )

    kwargs = {
        k: v
        for k, v in candidate_kwargs.items()
        if k in signature.parameters
    }

    model = SentenceEncoder(
        **kwargs
    ).to(device)

    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model, config


def build_test_loader(config):
    raw = load_from_disk(
        config["data_path"]
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            config["model_path"],
            local_files_only=True,
        )
    )

    test_ds = STSDataset(
        raw["test"],
        limit=config.get(
            "limit_test",
            None,
        ),
    )

    collator = STSCollator(
        tokenizer,
        max_length=config.get(
            "max_length",
            128,
        ),
    )

    loader = DataLoader(
        test_ds,
        batch_size=config.get(
            "batch_size",
            4,
        ),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    return loader, tokenizer


def pool_hidden(
    model,
    hidden,
    attention_mask,
):
    signature = inspect.signature(
        model.pool.forward
    )

    if (
        "attention_mask"
        in signature.parameters
    ):
        return model.pool(
            hidden,
            attention_mask=attention_mask,
        )

    return model.pool(
        hidden,
        attention_mask,
    )


# =========================================================
# Vectorized single-token knockout
# =========================================================

@torch.no_grad()
def knockout_one_sentence(
    model,
    hidden,
    attention_mask,
    other_embedding,
    target,
    baseline_pred,
    pair_id,
    side,
    input_ids,
    tokenizer,
):
    """
    hidden:
        [1, T, D]

    Knock out every content token one at a time.

    RoBERTa:
        <s> content ... </s> PAD

    Special tokens are left unchanged.
    """

    valid_len = int(
        attention_mask
        .sum()
        .item()
    )

    # positions:
    # 0            -> <s>
    # valid_len-1  -> </s>
    positions = list(
        range(
            1,
            valid_len - 1,
        )
    )

    content_len = len(
        positions
    )

    if content_len <= 0:
        return []

    # Make N copies of the same sentence.
    # Copy j will have content token j zeroed.
    variants = hidden.repeat(
        content_len,
        1,
        1,
    )

    row_ids = torch.arange(
        content_len,
        device=hidden.device,
    )

    pos_tensor = torch.tensor(
        positions,
        device=hidden.device,
        dtype=torch.long,
    )

    variants[
        row_ids,
        pos_tensor,
        :,
    ] = 0.0

    repeated_mask = (
        attention_mask.repeat(
            content_len,
            1,
        )
    )

    knocked_embedding = pool_hidden(
        model,
        variants,
        repeated_mask,
    )

    other = other_embedding.expand(
        content_len,
        -1,
    )

    if side == 1:
        knocked_pred = (
            F.cosine_similarity(
                knocked_embedding,
                other,
                dim=-1,
            )
        )
    else:
        knocked_pred = (
            F.cosine_similarity(
                other,
                knocked_embedding,
                dim=-1,
            )
        )

    target_vec = target.expand(
        content_len
    )

    baseline_vec = baseline_pred.expand(
        content_len
    )

    baseline_sqerr = (
        baseline_vec
        - target_vec
    ) ** 2

    knockout_sqerr = (
        knocked_pred
        - target_vec
    ) ** 2

    abs_sqerr_change = (
        knockout_sqerr
        - baseline_sqerr
    ).abs()

    abs_pred_change = (
        knocked_pred
        - baseline_vec
    ).abs()

    rows = []

    for rank, pos in enumerate(
        positions
    ):
        token_id = int(
            input_ids[pos].item()
        )

        token_text = (
            tokenizer.convert_ids_to_tokens(
                token_id
            )
        )

        # normalized position in [0,1]
        if content_len == 1:
            relative_position = 0.5
        else:
            relative_position = (
                rank
                / (content_len - 1)
            )

        # Five coarse relative-position bins.
        position_bin = min(
            4,
            int(
                rank
                * 5
                / content_len
            ),
        )

        rows.append({
            "pair_id":
                int(pair_id),

            "side":
                int(side),

            "content_pos":
                int(rank),

            "absolute_token_pos":
                int(pos),

            "content_len":
                int(content_len),

            "relative_position":
                float(relative_position),

            "position_bin":
                int(position_bin),

            "token_id":
                token_id,

            "token_text":
                token_text,

            "gold":
                float(target.item()),

            "baseline_pred":
                float(
                    baseline_pred.item()
                ),

            "knockout_pred":
                float(
                    knocked_pred[
                        rank
                    ].item()
                ),

            "baseline_sqerr":
                float(
                    baseline_sqerr[
                        rank
                    ].item()
                ),

            "knockout_sqerr":
                float(
                    knockout_sqerr[
                        rank
                    ].item()
                ),

            "abs_sqerr_change":
                float(
                    abs_sqerr_change[
                        rank
                    ].item()
                ),

            "abs_pred_change":
                float(
                    abs_pred_change[
                        rank
                    ].item()
                ),
        })

    return rows


# =========================================================
# One model / one seed
# =========================================================

@torch.no_grad()
def run_one(
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

    loader, tokenizer = (
        build_test_loader(
            config
        )
    )

    with open(
        run_dir / "metrics.json"
    ) as f:
        stored_metrics = json.load(f)

    all_rows = []

    full_preds = []
    full_gold = []

    global_pair_id = 0
    total_tokens = 0

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

        batch_size = (
            target.size(0)
        )

        # ---------------------------------------------
        # Backbone once
        # ---------------------------------------------

        h1 = model.backbone(
            **tok1
        ).last_hidden_state

        h2 = model.backbone(
            **tok2
        ).last_hidden_state

        # ---------------------------------------------
        # Normal prediction
        # ---------------------------------------------

        z1 = pool_hidden(
            model,
            h1,
            tok1["attention_mask"],
        )

        z2 = pool_hidden(
            model,
            h2,
            tok2["attention_mask"],
        )

        baseline_pred = (
            F.cosine_similarity(
                z1,
                z2,
                dim=-1,
            )
        )

        full_preds.extend(
            baseline_pred
            .detach()
            .cpu()
            .tolist()
        )

        full_gold.extend(
            target
            .detach()
            .cpu()
            .tolist()
        )

        # ---------------------------------------------
        # One token at a time
        # ---------------------------------------------

        for j in range(
            batch_size
        ):
            pair_id = (
                global_pair_id
                + j
            )

            rows1 = (
                knockout_one_sentence(
                    model=model,

                    hidden=h1[
                        j:j+1
                    ],

                    attention_mask=(
                        tok1[
                            "attention_mask"
                        ][j:j+1]
                    ),

                    other_embedding=(
                        z2[j:j+1]
                    ),

                    target=target[
                        j:j+1
                    ],

                    baseline_pred=(
                        baseline_pred[
                            j:j+1
                        ]
                    ),

                    pair_id=pair_id,

                    side=1,

                    input_ids=(
                        tok1[
                            "input_ids"
                        ][j]
                    ),

                    tokenizer=tokenizer,
                )
            )

            rows2 = (
                knockout_one_sentence(
                    model=model,

                    hidden=h2[
                        j:j+1
                    ],

                    attention_mask=(
                        tok2[
                            "attention_mask"
                        ][j:j+1]
                    ),

                    other_embedding=(
                        z1[j:j+1]
                    ),

                    target=target[
                        j:j+1
                    ],

                    baseline_pred=(
                        baseline_pred[
                            j:j+1
                        ]
                    ),

                    pair_id=pair_id,

                    side=2,

                    input_ids=(
                        tok2[
                            "input_ids"
                        ][j]
                    ),

                    tokenizer=tokenizer,
                )
            )

            for row in rows1:
                row["model"] = model_name
                row["seed"] = seed

            for row in rows2:
                row["model"] = model_name
                row["seed"] = seed

            all_rows.extend(
                rows1
            )

            all_rows.extend(
                rows2
            )

            total_tokens += (
                len(rows1)
                + len(rows2)
            )

        global_pair_id += (
            batch_size
        )

        if (
            batch_idx % 50 == 0
            or batch_idx == len(loader)
        ):
            print(
                f"batch "
                f"{batch_idx}/{len(loader)} "
                f"| token knockouts "
                f"{total_tokens}"
            )

    # =====================================================
    # Baseline reproduction
    # =====================================================

    full_sp, full_pr = (
        correlation_metrics(
            full_preds,
            full_gold,
        )
    )

    stored_sp = float(
        stored_metrics[
            "test_spearman"
        ]
    )

    stored_pr = float(
        stored_metrics[
            "test_pearson"
        ]
    )

    sp_diff = (
        full_sp - stored_sp
    )

    pr_diff = (
        full_pr - stored_pr
    )

    print()
    print(
        "Full-test baseline sanity:"
    )

    print(
        f"computed Spearman = "
        f"{full_sp:.6f}"
    )

    print(
        f"stored   Spearman = "
        f"{stored_sp:.6f}"
    )

    print(
        f"difference        = "
        f"{sp_diff:+.8f}"
    )

    print(
        f"computed Pearson  = "
        f"{full_pr:.6f}"
    )

    print(
        f"stored   Pearson  = "
        f"{stored_pr:.6f}"
    )

    print(
        f"difference        = "
        f"{pr_diff:+.8f}"
    )

    if (
        abs(sp_diff) > 1e-5
        or abs(pr_diff) > 1e-5
    ):
        raise RuntimeError(
            "Baseline reproduction failed."
        )

    print(
        "BASELINE REPRODUCTION PASSED"
    )

    # =====================================================
    # Save token-level responses
    # =====================================================

    token_df = pd.DataFrame(
        all_rows
    )

    token_df.to_csv(
        seed_dir
        / "per_token_responses.csv",
        index=False,
    )

    seed_summary = {
        "model":
            model_name,

        "seed":
            seed,

        "test_pairs":
            len(full_preds),

        "token_knockouts":
            len(token_df),

        "mean_abs_sqerr_change":
            float(
                token_df[
                    "abs_sqerr_change"
                ].mean()
            ),

        "median_abs_sqerr_change":
            float(
                token_df[
                    "abs_sqerr_change"
                ].median()
            ),

        "mean_abs_pred_change":
            float(
                token_df[
                    "abs_pred_change"
                ].mean()
            ),

        "baseline_spearman":
            full_sp,

        "baseline_pearson":
            full_pr,
    }

    with open(
        seed_dir / "summary.json",
        "w",
    ) as f:
        json.dump(
            seed_summary,
            f,
            indent=2,
        )

    with open(
        seed_dir / "status.json",
        "w",
    ) as f:
        json.dump(
            {
                "status":
                    "SUCCESS",
                **seed_summary,
            },
            f,
            indent=2,
        )

    success.touch()

    print()
    print(
        "token knockouts:",
        len(token_df),
    )

    print(
        "mean |Δ squared error|:",
        f"{seed_summary['mean_abs_sqerr_change']:.6f}",
    )

    print(
        "mean |Δ prediction|:",
        f"{seed_summary['mean_abs_pred_change']:.6f}",
    )

    del model
    torch.cuda.empty_cache()


# =========================================================
# Cross-seed analysis
# =========================================================

def aggregate_results(
    output_dir,
):
    files = list(
        output_dir.glob(
            "*/seed_*/per_token_responses.csv"
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
    # Average aligned token response across seeds.
    # -----------------------------------------------------

    aligned = (
        df
        .groupby(
            [
                "model",
                "pair_id",
                "side",
                "content_pos",
                "absolute_token_pos",
                "content_len",
                "position_bin",
                "token_id",
                "token_text",
            ],
            dropna=False,
        )
        .agg(
            aligned_seed_count=(
                "seed",
                "count",
            ),

            relative_position=(
                "relative_position",
                "first",
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
        summary_dir
        / "aligned_seed_token_responses.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Paper-style within-sentence positional heterogeneity.
    #
    # Paper:
    # average response across aligned seeds first,
    # then SD across positions within each sequence.
    # -----------------------------------------------------

    def population_std(series):
        return float(
            np.std(
                series.to_numpy(
                    dtype=np.float64
                ),
                ddof=0,
            )
        )

    sentence_heterogeneity = (
        aligned
        .groupby(
            [
                "model",
                "pair_id",
                "side",
            ]
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
        )
        .reset_index()
    )

    sentence_heterogeneity.to_csv(
        summary_dir
        / "sentence_positional_heterogeneity.csv",
        index=False,
    )

    heterogeneity_summary = (
        sentence_heterogeneity
        .groupby("model")
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

            std_positional_std=(
                "positional_response_std",
                "std",
            ),

            mean_token_response=(
                "mean_token_response",
                "mean",
            ),

            mean_max_token_response=(
                "max_token_response",
                "mean",
            ),
        )
        .reset_index()
    )

    heterogeneity_summary.to_csv(
        summary_dir
        / "heterogeneity_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Relative-position profile
    # 0 = beginning ... 4 = end
    # -----------------------------------------------------

    position_profile = (
        aligned
        .groupby(
            [
                "model",
                "position_bin",
            ]
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

    position_profile[
        "position_region"
    ] = position_profile[
        "position_bin"
    ].map({
        0: "start",
        1: "early",
        2: "middle",
        3: "late",
        4: "end",
    })

    position_profile.to_csv(
        summary_dir
        / "relative_position_profile.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Direct matched FLaG vs E3 token comparison
    # -----------------------------------------------------

    flag = aligned[
        aligned["model"] == "FLaG"
    ].copy()

    e3 = aligned[
        aligned["model"]
        == "E3_STFT_w8_h4_rect"
    ].copy()

    if (
        len(flag) > 0
        and len(e3) > 0
    ):
        merge_keys = [
            "pair_id",
            "side",
            "content_pos",
        ]

        comparison = flag[
            merge_keys
            + [
                "content_len",
                "position_bin",
                "token_id",
                "token_text",
                "abs_sqerr_change",
                "abs_pred_change",
            ]
        ].merge(
            e3[
                merge_keys
                + [
                    "abs_sqerr_change",
                    "abs_pred_change",
                ]
            ],
            on=merge_keys,
            suffixes=(
                "_flag",
                "_e3",
            ),
            how="inner",
        )

        comparison[
            "e3_minus_flag_abs_sqerr"
        ] = (
            comparison[
                "abs_sqerr_change_e3"
            ]
            - comparison[
                "abs_sqerr_change_flag"
            ]
        )

        comparison[
            "e3_minus_flag_abs_pred"
        ] = (
            comparison[
                "abs_pred_change_e3"
            ]
            - comparison[
                "abs_pred_change_flag"
            ]
        )

        comparison.to_csv(
            summary_dir
            / "flag_vs_e3_token_comparison.csv",
            index=False,
        )

        position_difference = (
            comparison
            .groupby(
                "position_bin"
            )
            .agg(
                tokens=(
                    "content_pos",
                    "count",
                ),

                mean_e3_minus_flag_sqerr=(
                    "e3_minus_flag_abs_sqerr",
                    "mean",
                ),

                mean_e3_minus_flag_pred=(
                    "e3_minus_flag_abs_pred",
                    "mean",
                ),
            )
            .reset_index()
        )

        position_difference[
            "position_region"
        ] = position_difference[
            "position_bin"
        ].map({
            0: "start",
            1: "early",
            2: "middle",
            3: "late",
            4: "end",
        })

        position_difference.to_csv(
            summary_dir
            / "flag_vs_e3_position_difference.csv",
            index=False,
        )

    print()
    print("=" * 72)
    print(
        "PAPER-STYLE POSITIONAL HETEROGENEITY"
    )
    print("=" * 72)

    print(
        heterogeneity_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print("=" * 72)
    print(
        "RELATIVE POSITION PROFILE"
    )
    print("=" * 72)

    print(
        position_profile.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    if (
        len(flag) > 0
        and len(e3) > 0
    ):
        print()
        print("=" * 72)
        print(
            "E3 - FLaG POSITION DIFFERENCE"
        )
        print(
            "positive = E3 more sensitive"
        )
        print("=" * 72)

        print(
            position_difference.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.6f}"
                ),
            )
        )

    print()
    print(
        "Saved summaries to:",
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
        / "E7_token_knockout"
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

    aggregate_results(
        output_dir
    )


if __name__ == "__main__":
    main()
