#!/usr/bin/env python3
"""Small numerical sanity check for the exact masked DC-removal operation."""

from pathlib import Path
import sys

import torch

# Make the repository importable when this file is executed directly as:
#   python text_repro/check_exact_dc_removal.py
# train_sts imports AMPCliff.factory..., so Python needs the directory that
# CONTAINS the AMPCliff repository, not just the repository itself.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = _REPO_ROOT.parent
for _p in (_REPO_PARENT, _REPO_ROOT / "text_repro"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

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
