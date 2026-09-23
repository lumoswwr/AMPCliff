import argparse
import json
from pathlib import Path

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


@torch.no_grad()
def pool_with_length(
    model,
    hidden,
    valid_len,
    target_len,
):
    """
    hidden: [1, batch_T, D]

    Keep only valid hidden states, then zero-pad to target_len.
    """

    x = hidden[
        :,
        :valid_len,
        :
    ]

    _, _, D = x.shape

    if target_len < valid_len:
        raise ValueError(
            f"target_len={target_len} "
            f"< valid_len={valid_len}"
        )

    padded = x.new_zeros(
        1,
        target_len,
        D,
    )

    padded[
        :,
        :valid_len,
        :
    ] = x

    mask = torch.zeros(
        1,
        target_len,
        dtype=torch.long,
        device=x.device,
    )

    mask[
        :,
        :valid_len,
    ] = 1

    return pool_hidden(
        model,
        padded,
        mask,
    )


@torch.no_grad()
def evaluate_one(
    model_name,
    seed,
    run_dir,
    device,
):
    print()
    print("=" * 76)
    print(
        model_name,
        "| seed",
        seed,
    )
    print("=" * 76)

    seed_everything(seed)

    model, config = build_model(
        run_dir,
        device,
    )

    loader, _ = build_test_loader(
        config
    )

    modes = {
        "normal": [],
        "pair_exact": [],
        "sentence_exact": [],
        "fixed_128": [],
    }

    gold = []

    pair_id = 0

    per_pair_rows = []

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

        # ---------------------------------------------
        # Backbone once.
        # ---------------------------------------------

        h1 = model.backbone(
            **tok1
        ).last_hidden_state

        h2 = model.backbone(
            **tok2
        ).last_hidden_state

        # ---------------------------------------------
        # Original batch-padded inference.
        # ---------------------------------------------

        z1_normal = pool_hidden(
            model,
            h1,
            tok1["attention_mask"],
        )

        z2_normal = pool_hidden(
            model,
            h2,
            tok2["attention_mask"],
        )

        pred_normal = (
            F.cosine_similarity(
                z1_normal,
                z2_normal,
                dim=-1,
            )
        )

        # ---------------------------------------------
        # Remaining modes are evaluated per pair.
        # ---------------------------------------------

        for j in range(B):

            L1 = int(
                tok1[
                    "attention_mask"
                ][j].sum().item()
            )

            L2 = int(
                tok2[
                    "attention_mask"
                ][j].sum().item()
            )

            pair_len = max(
                L1,
                L2,
            )

            hj1 = h1[
                j:j+1
            ]

            hj2 = h2[
                j:j+1
            ]

            # pair_exact:
            # both sentences use same length = max(L1,L2)
            z1_pair = pool_with_length(
                model,
                hj1,
                L1,
                pair_len,
            )

            z2_pair = pool_with_length(
                model,
                hj2,
                L2,
                pair_len,
            )

            pred_pair = (
                F.cosine_similarity(
                    z1_pair,
                    z2_pair,
                    dim=-1,
                )[0]
            )

            # sentence_exact:
            # each sentence uses its own exact valid length
            z1_exact = pool_with_length(
                model,
                hj1,
                L1,
                L1,
            )

            z2_exact = pool_with_length(
                model,
                hj2,
                L2,
                L2,
            )

            pred_exact = (
                F.cosine_similarity(
                    z1_exact,
                    z2_exact,
                    dim=-1,
                )[0]
            )

            # fixed_128:
            # deterministic global frequency grid
            z1_fixed = pool_with_length(
                model,
                hj1,
                L1,
                128,
            )

            z2_fixed = pool_with_length(
                model,
                hj2,
                L2,
                128,
            )

            pred_fixed = (
                F.cosine_similarity(
                    z1_fixed,
                    z2_fixed,
                    dim=-1,
                )[0]
            )

            pn = float(
                pred_normal[
                    j
                ].item()
            )

            pp = float(
                pred_pair.item()
            )

            pe = float(
                pred_exact.item()
            )

            pf = float(
                pred_fixed.item()
            )

            modes[
                "normal"
            ].append(pn)

            modes[
                "pair_exact"
            ].append(pp)

            modes[
                "sentence_exact"
            ].append(pe)

            modes[
                "fixed_128"
            ].append(pf)

            y = float(
                target[j].item()
            )

            gold.append(y)

            per_pair_rows.append({
                "pair_id":
                    pair_id,

                "len1":
                    L1,

                "len2":
                    L2,

                "pair_max_len":
                    pair_len,

                "gold":
                    y,

                "normal":
                    pn,

                "pair_exact":
                    pp,

                "sentence_exact":
                    pe,

                "fixed_128":
                    pf,

                "normal_to_pair_drift":
                    abs(
                        pp - pn
                    ),

                "pair_to_sentence_drift":
                    abs(
                        pe - pp
                    ),

                "normal_to_sentence_drift":
                    abs(
                        pe - pn
                    ),

                "normal_to_fixed128_drift":
                    abs(
                        pf - pn
                    ),
            })

            pair_id += 1

        if (
            batch_idx % 100 == 0
            or batch_idx == len(loader)
        ):
            print(
                f"batch "
                f"{batch_idx}/{len(loader)}"
            )

    # =====================================================
    # Metrics
    # =====================================================

    rows = []

    for mode, pred in modes.items():

        sp, pr = correlation_metrics(
            pred,
            gold,
        )

        rows.append({
            "model":
                model_name,

            "seed":
                seed,

            "mode":
                mode,

            "spearman":
                sp,

            "pearson":
                pr,
        })

    result = pd.DataFrame(rows)

    normal_sp = float(
        result.loc[
            result["mode"]
            == "normal",
            "spearman",
        ].iloc[0]
    )

    normal_pr = float(
        result.loc[
            result["mode"]
            == "normal",
            "pearson",
        ].iloc[0]
    )

    result[
        "spearman_vs_normal"
    ] = (
        result["spearman"]
        - normal_sp
    )

    result[
        "pearson_vs_normal"
    ] = (
        result["pearson"]
        - normal_pr
    )

    pair_df = pd.DataFrame(
        per_pair_rows
    )

    print()
    print(
        result.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()
    print(
        "Mean prediction drift:"
    )

    for col in [
        "normal_to_pair_drift",
        "pair_to_sentence_drift",
        "normal_to_sentence_drift",
        "normal_to_fixed128_drift",
    ]:
        print(
            f"{col:30s} "
            f"{pair_df[col].mean():.8f}"
        )

    return result, pair_df


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
        default=[0],
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
        / "E10_padding_conventions"
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

    all_results = []

    for model_key in args.models:

        spec = specs[
            model_key
        ]

        for seed in args.seeds:

            result, pair_df = evaluate_one(
                model_name=(
                    spec["name"]
                ),

                seed=seed,

                run_dir=(
                    spec["run_root"]
                    / f"seed_{seed}"
                ),

                device=device,
            )

            all_results.append(
                result
            )

            run_out = (
                output_dir
                / spec["name"]
                / f"seed_{seed}"
            )

            run_out.mkdir(
                parents=True,
                exist_ok=True,
            )

            result.to_csv(
                run_out
                / "metrics.csv",
                index=False,
            )

            pair_df.to_csv(
                run_out
                / "per_pair_predictions.csv",
                index=False,
            )

    result_all = pd.concat(
        all_results,
        ignore_index=True,
    )

    result_all.to_csv(
        output_dir
        / "all_metrics.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
