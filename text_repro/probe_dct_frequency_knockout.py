import argparse
import inspect
import json
import math
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
# Utilities
# =========================================================

def correlation_metrics(pred, gold):
    pred = np.asarray(pred, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.float64)

    sp = float(
        spearmanr(pred, gold)[0]
    )

    pr = float(
        pearsonr(pred, gold)[0]
    )

    return sp, pr


# =========================================================
# Paper-style geometric DCT bands
# =========================================================

def geometric_band_sizes(
    length,
    num_bands=8,
    base=4.0,
):
    """
    Paper-style allocation:

    1. every band receives at least one coefficient
    2. remaining coefficients are allocated
       proportional to base ** b
    3. largest-remainder rounding preserves total length

    Examples from the FLaG paper:
        T=9:
        [1,1,1,1,1,1,1,2]

        T=25:
        [1,1,1,1,1,2,4,14]
    """

    if length < num_bands:
        raise ValueError(
            f"length={length} < num_bands={num_bands}"
        )

    sizes = np.ones(
        num_bands,
        dtype=np.int64,
    )

    remaining = (
        length - num_bands
    )

    if remaining == 0:
        return sizes.tolist()

    weights = np.array(
        [
            base ** b
            for b in range(num_bands)
        ],
        dtype=np.float64,
    )

    raw = (
        remaining
        * weights
        / weights.sum()
    )

    extra = np.floor(
        raw
    ).astype(np.int64)

    sizes += extra

    leftover = int(
        remaining - extra.sum()
    )

    if leftover > 0:
        fractional = (
            raw - extra
        )

        order = np.argsort(
            -fractional,
            kind="stable",
        )

        for idx in order[:leftover]:
            sizes[idx] += 1

    assert sizes.sum() == length
    assert np.all(sizes >= 1)

    return sizes.tolist()


def geometric_band_slice(
    length,
    band,
    num_bands=8,
    base=4.0,
):
    sizes = geometric_band_sizes(
        length=length,
        num_bands=num_bands,
        base=base,
    )

    start = sum(
        sizes[:band]
    )

    end = (
        start
        + sizes[band]
    )

    return start, end


def verify_band_partition():
    t9 = geometric_band_sizes(9)
    t25 = geometric_band_sizes(25)

    expected9 = [
        1, 1, 1, 1,
        1, 1, 1, 2,
    ]

    expected25 = [
        1, 1, 1, 1,
        1, 2, 4, 14,
    ]

    print(
        "T=9  bands:",
        t9,
    )

    print(
        "T=25 bands:",
        t25,
    )

    assert t9 == expected9
    assert t25 == expected25

    print(
        "GEOMETRIC BAND TEST PASSED"
    )


# =========================================================
# Orthonormal DCT-II
# =========================================================

_DCT_CACHE = {}


def get_dct_matrix(
    length,
    device,
    dtype,
):
    key = (
        length,
        str(device),
        str(dtype),
    )

    if key in _DCT_CACHE:
        return _DCT_CACHE[key]

    n = torch.arange(
        length,
        device=device,
        dtype=dtype,
    )

    k = torch.arange(
        length,
        device=device,
        dtype=dtype,
    ).unsqueeze(1)

    matrix = torch.cos(
        math.pi
        / length
        * (n + 0.5)
        * k
    )

    scale = torch.full(
        (length,),
        math.sqrt(
            2.0 / length
        ),
        device=device,
        dtype=dtype,
    )

    scale[0] = math.sqrt(
        1.0 / length
    )

    matrix = (
        scale[:, None]
        * matrix
    )

    _DCT_CACHE[key] = matrix

    return matrix


def verify_dct():
    torch.manual_seed(0)

    x = torch.randn(
        17,
        8,
    )

    c = get_dct_matrix(
        length=17,
        device=x.device,
        dtype=x.dtype,
    )

    z = c @ x
    recon = c.transpose(0, 1) @ z

    error = (
        recon - x
    ).abs().max().item()

    print(
        "DCT reconstruction error:",
        error,
    )

    assert error < 1e-5

    print(
        "DCT RECONSTRUCTION TEST PASSED"
    )


# =========================================================
# DCT knockout on final-layer hidden states
# =========================================================

def dct_band_knockout(
    hidden,
    attention_mask,
    band,
):
    """
    hidden:
        [B, T, D]

    RoBERTa single-sentence layout:
        <s> content tokens </s> PAD ...

    We leave <s> and </s> unchanged and apply DCT only
    to the content-token hidden states.

    All samples passed here must have >=8 content tokens.
    """

    result = hidden.clone()

    valid_lengths = (
        attention_mask
        .long()
        .sum(dim=1)
    )

    for i in range(hidden.size(0)):
        valid_len = int(
            valid_lengths[i].item()
        )

        # Exclude <s> and </s>.
        content_start = 1
        content_end = (
            valid_len - 1
        )

        content_len = (
            content_end
            - content_start
        )

        if content_len < 8:
            raise ValueError(
                "E6 paper-style 8-band probe received "
                f"content_len={content_len} < 8"
            )

        x = hidden[
            i,
            content_start:content_end,
            :,
        ]

        c = get_dct_matrix(
            length=content_len,
            device=x.device,
            dtype=x.dtype,
        )

        # [N,D]
        coeff = (
            c @ x
        )

        start, end = (
            geometric_band_slice(
                length=content_len,
                band=band,
                num_bands=8,
                base=4.0,
            )
        )

        coeff = coeff.clone()

        coeff[
            start:end,
            :
        ] = 0.0

        reconstructed = (
            c.transpose(0, 1)
            @ coeff
        )

        result[
            i,
            content_start:content_end,
            :,
        ] = reconstructed

    return result


# =========================================================
# Model / data
# =========================================================

def build_model(
    run_dir,
    device,
):
    with open(
        run_dir / "config.json"
    ) as f:
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
        key: value
        for key, value
        in candidate_kwargs.items()
        if key in signature.parameters
    }

    model = SentenceEncoder(
        **kwargs
    ).to(device)

    checkpoint = torch.load(
        run_dir / "best_model.pt",
        map_location=device,
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model, config


def build_test_loader(
    config,
):
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

    return loader


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
# One trained model / one seed
# =========================================================

@torch.no_grad()
def run_one(
    model_name,
    seed,
    run_dir,
    output_dir,
    bands,
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
        run_dir=run_dir,
        device=device,
    )

    loader = build_test_loader(
        config
    )

    with open(
        run_dir / "metrics.json"
    ) as f:
        stored_metrics = json.load(f)

    full_preds = []
    full_gold = []

    sample_rows = []

    total_pairs = 0
    eligible_pairs = 0

    global_pair_id = 0

    for batch_idx, (
        tok1,
        tok2,
        score,
    ) in enumerate(
        loader,
        start=1,
    ):
        tok1 = {
            key: value.to(device)
            for key, value
            in tok1.items()
        }

        tok2 = {
            key: value.to(device)
            for key, value
            in tok2.items()
        }

        target = (
            score.to(device)
            / 5.0
        )

        batch_size = (
            target.size(0)
        )

        pair_ids = torch.arange(
            global_pair_id,
            global_pair_id
            + batch_size,
            device=device,
        )

        global_pair_id += (
            batch_size
        )

        # ---------------------------------------------
        # Run RoBERTa exactly once for each sentence.
        # ---------------------------------------------

        h1 = model.backbone(
            **tok1
        ).last_hidden_state

        h2 = model.backbone(
            **tok2
        ).last_hidden_state

        # ---------------------------------------------
        # Unperturbed baseline
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

        total_pairs += (
            batch_size
        )

        # RoBERTa valid length includes:
        # <s> + content + </s>
        content_len1 = (
            tok1["attention_mask"]
            .sum(dim=1)
            .long()
            - 2
        )

        content_len2 = (
            tok2["attention_mask"]
            .sum(dim=1)
            .long()
            - 2
        )

        # Paper's K=8 partition requires T >= 8.
        eligible = (
            (content_len1 >= 8)
            & (content_len2 >= 8)
        )

        if not eligible.any():
            continue

        eligible_pairs += int(
            eligible.sum().item()
        )

        h1_e = h1[eligible]
        h2_e = h2[eligible]

        mask1_e = (
            tok1["attention_mask"][
                eligible
            ]
        )

        mask2_e = (
            tok2["attention_mask"][
                eligible
            ]
        )

        target_e = target[
            eligible
        ]

        baseline_e = (
            baseline_pred[
                eligible
            ]
        )

        ids_e = pair_ids[
            eligible
        ]

        len1_e = content_len1[
            eligible
        ]

        len2_e = content_len2[
            eligible
        ]

        baseline_sqerr = (
            baseline_e
            - target_e
        ) ** 2

        # ---------------------------------------------
        # Remove one DCT band from BOTH sentences.
        # ---------------------------------------------

        for band in bands:
            h1_knock = (
                dct_band_knockout(
                    h1_e,
                    mask1_e,
                    band=band,
                )
            )

            h2_knock = (
                dct_band_knockout(
                    h2_e,
                    mask2_e,
                    band=band,
                )
            )

            z1_knock = pool_hidden(
                model,
                h1_knock,
                mask1_e,
            )

            z2_knock = pool_hidden(
                model,
                h2_knock,
                mask2_e,
            )

            pred_knock = (
                F.cosine_similarity(
                    z1_knock,
                    z2_knock,
                    dim=-1,
                )
            )

            knock_sqerr = (
                pred_knock
                - target_e
            ) ** 2

            abs_sqerr_change = (
                knock_sqerr
                - baseline_sqerr
            ).abs()

            abs_pred_change = (
                pred_knock
                - baseline_e
            ).abs()

            for j in range(
                target_e.size(0)
            ):
                sample_rows.append({
                    "model":
                        model_name,

                    "seed":
                        seed,

                    "pair_id":
                        int(
                            ids_e[j].item()
                        ),

                    "band":
                        int(band),

                    "content_len1":
                        int(
                            len1_e[j].item()
                        ),

                    "content_len2":
                        int(
                            len2_e[j].item()
                        ),

                    "gold":
                        float(
                            target_e[j].item()
                        ),

                    "baseline_pred":
                        float(
                            baseline_e[j].item()
                        ),

                    "knockout_pred":
                        float(
                            pred_knock[j].item()
                        ),

                    "baseline_sqerr":
                        float(
                            baseline_sqerr[
                                j
                            ].item()
                        ),

                    "knockout_sqerr":
                        float(
                            knock_sqerr[
                                j
                            ].item()
                        ),

                    "abs_sqerr_change":
                        float(
                            abs_sqerr_change[
                                j
                            ].item()
                        ),

                    "abs_pred_change":
                        float(
                            abs_pred_change[
                                j
                            ].item()
                        ),
                })

        if (
            batch_idx % 100 == 0
        ):
            print(
                f"batch "
                f"{batch_idx}/{len(loader)}"
            )

    # =====================================================
    # Full-test baseline sanity check
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
            "Baseline reproduction failed. "
            "Do not interpret knockout results."
        )

    print(
        "BASELINE REPRODUCTION PASSED"
    )

    print()
    print(
        f"Paper-valid subset: "
        f"{eligible_pairs}/{total_pairs} "
        f"pairs "
        f"({eligible_pairs / total_pairs:.2%})"
    )

    # =====================================================
    # Save sample-level responses
    # =====================================================

    sample_df = pd.DataFrame(
        sample_rows
    )

    sample_df.to_csv(
        seed_dir
        / "per_sample_band_responses.csv",
        index=False,
    )

    # =====================================================
    # Per-band summary
    # =====================================================

    summary_rows = []

    for band in bands:
        band_df = (
            sample_df[
                sample_df["band"]
                == band
            ]
        )

        base_sp, base_pr = (
            correlation_metrics(
                band_df[
                    "baseline_pred"
                ],
                band_df["gold"],
            )
        )

        knock_sp, knock_pr = (
            correlation_metrics(
                band_df[
                    "knockout_pred"
                ],
                band_df["gold"],
            )
        )

        summary_rows.append({
            "model":
                model_name,

            "seed":
                seed,

            "band":
                band,

            "num_pairs":
                len(band_df),

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

            "mean_abs_sqerr_change":
                band_df[
                    "abs_sqerr_change"
                ].mean(),

            "median_abs_sqerr_change":
                band_df[
                    "abs_sqerr_change"
                ].median(),

            "mean_abs_pred_change":
                band_df[
                    "abs_pred_change"
                ].mean(),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        seed_dir
        / "band_summary.csv",
        index=False,
    )

    status = {
        "status":
            "SUCCESS",

        "model":
            model_name,

        "seed":
            seed,

        "total_test_pairs":
            total_pairs,

        "eligible_pairs":
            eligible_pairs,

        "eligible_fraction":
            eligible_pairs
            / total_pairs,

        "full_baseline_spearman":
            full_sp,

        "stored_spearman":
            stored_sp,

        "full_baseline_pearson":
            full_pr,

        "stored_pearson":
            stored_pr,

        "bands":
            bands,

        "probe":
            (
                "final-layer content-token "
                "DCT band knockout"
            ),

        "num_bands":
            8,

        "geometric_base":
            4.0,

        "special_tokens_perturbed":
            False,

        "pair_perturbation":
            "same band removed from both sentences",
    }

    with open(
        seed_dir / "status.json",
        "w",
    ) as f:
        json.dump(
            status,
            f,
            indent=2,
        )

    

    print()
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    del model
    torch.cuda.empty_cache()


# =========================================================
# Cross-seed aggregation
# =========================================================

def aggregate_results(
    output_dir,
):
    summary_files = list(
        output_dir.glob(
            "*/seed_*/band_summary.csv"
        )
    )

    sample_files = list(
        output_dir.glob(
            "*/seed_*/per_sample_band_responses.csv"
        )
    )

    if not summary_files:
        return

    summary_df = pd.concat(
        [
            pd.read_csv(path)
            for path in summary_files
        ],
        ignore_index=True,
    )

    summary_root = (
        output_dir / "summary"
    )

    summary_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        summary_root
        / "all_seed_band_results.csv",
        index=False,
    )

    cross_seed = (
        summary_df
        .groupby(
            [
                "model",
                "band",
            ]
        )
        .agg(
            seeds=(
                "seed",
                "count",
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

            std_pearson_drop=(
                "pearson_drop",
                "std",
            ),

            mean_abs_sqerr_change=(
                "mean_abs_sqerr_change",
                "mean",
            ),

            mean_abs_pred_change=(
                "mean_abs_pred_change",
                "mean",
            ),
        )
        .reset_index()
    )

    cross_seed.to_csv(
        summary_root
        / "cross_seed_band_summary.csv",
        index=False,
    )

    print()
    print("=" * 72)
    print("CROSS-SEED BAND SUMMARY")
    print("=" * 72)

    print(
        cross_seed.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    # -----------------------------------------------------
    # Paper-style aligned-seed averaging:
    # average response for each pair over aligned seeds,
    # then aggregate across pairs.
    # -----------------------------------------------------

    if sample_files:
        sample_df = pd.concat(
            [
                pd.read_csv(path)
                for path in sample_files
            ],
            ignore_index=True,
        )

        aligned = (
            sample_df
            .groupby(
                [
                    "model",
                    "band",
                    "pair_id",
                ]
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
            summary_root
            / "aligned_seed_pair_responses.csv",
            index=False,
        )

        aligned_band = (
            aligned
            .groupby(
                [
                    "model",
                    "band",
                ]
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

        aligned_band.to_csv(
            summary_root
            / "aligned_seed_band_response_summary.csv",
            index=False,
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
        type=int,
        nargs="+",
        default=[
            0,
            1,
            2,
        ],
    )

    parser.add_argument(
        "--bands",
        type=int,
        nargs="+",
        default=list(
            range(8)
        ),
    )

    parser.add_argument(
        "--root",
        default=(
            "/home/data/home/wwr_lumos/"
            "AMPCliff/outputs/text/stsbenchmark"
        ),
    )

    args = parser.parse_args()

    for band in args.bands:
        if not 0 <= band <= 7:
            raise ValueError(
                f"band must be 0..7, got {band}"
            )

    verify_band_partition()
    verify_dct()

    root = Path(
        args.root
    )

    specs = {
        "flag": {
            "name":
                "FLaG",

            "run_root":
                root
                / "FLaG",
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
        / "E6_dct_frequency_knockout"
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
                model_name=spec["name"],
                seed=seed,
                run_dir=(
                    spec["run_root"]
                    / f"seed_{seed}"
                ),
                output_dir=output_dir,
                bands=args.bands,
                device=device,
            )

    aggregate_results(
        output_dir
    )


if __name__ == "__main__":
    main()
