import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from text_repro.probe_token_knockout import (
    build_model,
    build_test_loader,
    pool_hidden,
)

from text_repro.train_sts import (
    seed_everything,
)


TARGET_LENGTHS = [
    16,
    24,
    32,
    48,
    64,
    128,
]


def pad_hidden(
    hidden_valid,
    target_len,
):
    """
    hidden_valid: [1,L,D]
    """

    _, L, D = hidden_valid.shape

    if target_len < L:
        raise ValueError(
            f"target_len={target_len} < L={L}"
        )

    padded = hidden_valid.new_zeros(
        1,
        target_len,
        D,
    )

    padded[
        :,
        :L,
        :,
    ] = hidden_valid

    mask = torch.zeros(
        1,
        target_len,
        dtype=torch.long,
        device=hidden_valid.device,
    )

    mask[
        :,
        :L,
    ] = 1

    return padded, mask


@torch.no_grad()
def run_one(
    model_key,
    model_name,
    seed,
    run_dir,
    output_dir,
    device,
    max_pairs,
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

    rows = []

    pair_id = 0
    kept_pairs = 0

    for (
        tok1,
        tok2,
        score,
    ) in loader:

        tok1 = {
            k: v.to(device)
            for k, v in tok1.items()
        }

        tok2 = {
            k: v.to(device)
            for k, v in tok2.items()
        }

        # Backbone only once.
        h1_batch = model.backbone(
            **tok1
        ).last_hidden_state

        h2_batch = model.backbone(
            **tok2
        ).last_hidden_state

        # Natural batch-padded pooling.
        natural_z1 = pool_hidden(
            model,
            h1_batch,
            tok1["attention_mask"],
        )

        natural_z2 = pool_hidden(
            model,
            h2_batch,
            tok2["attention_mask"],
        )

        natural_pred = (
            F.cosine_similarity(
                natural_z1,
                natural_z2,
                dim=-1,
            )
        )

        B = h1_batch.size(0)

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

            pair_max_len = max(
                L1,
                L2,
            )

            # Focus on the short regime where E3
            # previously showed its clearest gain.
            if pair_max_len > 16:
                pair_id += 1
                continue

            h1 = (
                h1_batch[
                    j:j+1,
                    :L1,
                    :
                ].clone()
            )

            h2 = (
                h2_batch[
                    j:j+1,
                    :L2,
                    :
                ].clone()
            )

            # -----------------------------------------
            # Exact-length reference
            # -----------------------------------------

            m1_exact = torch.ones(
                1,
                L1,
                dtype=torch.long,
                device=device,
            )

            m2_exact = torch.ones(
                1,
                L2,
                dtype=torch.long,
                device=device,
            )

            z1_exact = pool_hidden(
                model,
                h1,
                m1_exact,
            )

            z2_exact = pool_hidden(
                model,
                h2,
                m2_exact,
            )

            pred_exact = (
                F.cosine_similarity(
                    z1_exact,
                    z2_exact,
                    dim=-1,
                )[0]
            )

            nat_pred = (
                natural_pred[j]
            )

            natural_vs_exact = float(
                (
                    nat_pred
                    - pred_exact
                ).abs().item()
            )

            # -----------------------------------------
            # Controlled artificial padding
            # -----------------------------------------

            for target_len in TARGET_LENGTHS:

                if target_len < pair_max_len:
                    continue

                h1_pad, m1_pad = (
                    pad_hidden(
                        h1,
                        target_len,
                    )
                )

                h2_pad, m2_pad = (
                    pad_hidden(
                        h2,
                        target_len,
                    )
                )

                z1_pad = pool_hidden(
                    model,
                    h1_pad,
                    m1_pad,
                )

                z2_pad = pool_hidden(
                    model,
                    h2_pad,
                    m2_pad,
                )

                pred_pad = (
                    F.cosine_similarity(
                        z1_pad,
                        z2_pad,
                        dim=-1,
                    )[0]
                )

                cos1 = (
                    F.cosine_similarity(
                        z1_exact,
                        z1_pad,
                        dim=-1,
                    )[0]
                )

                cos2 = (
                    F.cosine_similarity(
                        z2_exact,
                        z2_pad,
                        dim=-1,
                    )[0]
                )

                rows.append({
                    "model":
                        model_name,

                    "seed":
                        seed,

                    "pair_id":
                        pair_id,

                    "len1":
                        L1,

                    "len2":
                        L2,

                    "pair_max_len":
                        pair_max_len,

                    "target_len":
                        target_len,

                    "exact_pred":
                        float(
                            pred_exact.item()
                        ),

                    "padded_pred":
                        float(
                            pred_pad.item()
                        ),

                    "abs_pred_drift":
                        float(
                            (
                                pred_pad
                                - pred_exact
                            ).abs().item()
                        ),

                    "embedding_cosine_loss_s1":
                        float(
                            (
                                1.0 - cos1
                            ).item()
                        ),

                    "embedding_cosine_loss_s2":
                        float(
                            (
                                1.0 - cos2
                            ).item()
                        ),

                    "natural_vs_exact_pred_drift":
                        natural_vs_exact,
                })

            kept_pairs += 1
            pair_id += 1

            if (
                max_pairs is not None
                and kept_pairs >= max_pairs
            ):
                break

        if (
            max_pairs is not None
            and kept_pairs >= max_pairs
        ):
            break

    df = pd.DataFrame(rows)

    out_dir = (
        output_dir
        / model_name
        / f"seed_{seed}"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out_dir
        / "padding_invariance.csv",
        index=False,
    )

    summary = (
        df
        .groupby(
            "target_len"
        )
        .agg(
            pairs=(
                "pair_id",
                "count",
            ),

            mean_abs_pred_drift=(
                "abs_pred_drift",
                "mean",
            ),

            median_abs_pred_drift=(
                "abs_pred_drift",
                "median",
            ),

            max_abs_pred_drift=(
                "abs_pred_drift",
                "max",
            ),

            mean_embedding_cosine_loss_s1=(
                "embedding_cosine_loss_s1",
                "mean",
            ),

            mean_embedding_cosine_loss_s2=(
                "embedding_cosine_loss_s2",
                "mean",
            ),

            mean_natural_vs_exact_pred_drift=(
                "natural_vs_exact_pred_drift",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        out_dir
        / "summary.csv",
        index=False,
    )

    print()
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.8f}"
            ),
        )
    )

    del model
    torch.cuda.empty_cache()


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
        "--max_pairs",
        type=int,
        default=200,
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
        / "E9_padding_invariance"
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
                model_name=spec["name"],
                seed=seed,
                run_dir=(
                    spec["run_root"]
                    / f"seed_{seed}"
                ),
                output_dir=output_dir,
                device=device,
                max_pairs=args.max_pairs,
            )


if __name__ == "__main__":
    main()
