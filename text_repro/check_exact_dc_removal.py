#!/usr/bin/env python3
"""Small numerical sanity check for the exact masked DC-removal operation."""

import torch

from train_sts import remove_dc_component


def main():
    torch.manual_seed(0)

    x = torch.randn(4, 11, 7)
    lengths = torch.tensor([11, 8, 5, 2])

    positions = torch.arange(x.size(1))[None, :]
    mask = (positions < lengths[:, None]).long()

    centered = remove_dc_component(x, mask)

    valid_sum = (
        centered
        * mask.unsqueeze(-1).to(centered.dtype)
    ).sum(dim=1)

    max_abs_dc = float(valid_sum.abs().max())

    padded = centered[
        ~mask.bool()
    ]
    max_abs_padding = (
        float(padded.abs().max())
        if padded.numel()
        else 0.0
    )

    print("max |valid-token DC sum|:", max_abs_dc)
    print("max |padding value|:", max_abs_padding)

    if max_abs_dc > 1e-5:
        raise RuntimeError("DC removal sanity check failed.")
    if max_abs_padding > 1e-7:
        raise RuntimeError("Padding-zero sanity check failed.")

    print("PASS: exact masked DC removal is numerically zero-sum.")


if __name__ == "__main__":
    main()
