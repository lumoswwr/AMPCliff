#!/usr/bin/env python3
"""Numerically verify exact No-DC and DC-only spectral controls."""

from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = _REPO_ROOT.parent
for _p in (_REPO_PARENT, _REPO_ROOT / "text_repro"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from AMPCliff.factory.pooling.flag_pooling import (
    FFTLatentAttentionGatePooling,
)


def make_pool(*, remove_dc=False, dc_only=False):
    return FFTLatentAttentionGatePooling(
        d_model=8,
        num_latents=2,
        num_heads=2,
        dropout=0.0,
        remove_dc=remove_dc,
        dc_only=dc_only,
    )


def main():
    torch.manual_seed(0)

    # Variable valid lengths inside one dynamically padded batch.
    x = torch.randn(4, 11, 8)
    lengths = torch.tensor([11, 8, 5, 2])
    positions = torch.arange(x.size(1))[None, :]
    mask = (positions < lengths[:, None]).long()

    base = make_pool()
    nodc = make_pool(remove_dc=True)
    dconly = make_pool(dc_only=True)

    # Learned parameters are irrelevant here. We inspect the analysis FFT.
    freq_base = base._to_frequency_tokens(
        x,
        attention_mask=mask,
    )
    freq_nodc = nodc._to_frequency_tokens(
        x,
        attention_mask=mask,
    )
    freq_dconly = dconly._to_frequency_tokens(
        x,
        attention_mask=mask,
    )

    original_dc_magnitude = float(
        freq_base[:, 0, :].abs().max()
    )

    # No-DC: k=0 must be zero, all k>0 unchanged.
    nodc_dc = float(
        freq_nodc[:, 0, :].abs().max()
    )
    nodc_non_dc_change = float(
        (
            freq_nodc[:, 1:, :]
            - freq_base[:, 1:, :]
        ).abs().max()
    )

    # DC-only: k=0 must be unchanged, all k>0 must be zero.
    dconly_dc_change = float(
        (
            freq_dconly[:, 0, :]
            - freq_base[:, 0, :]
        ).abs().max()
    )
    dconly_non_dc = float(
        freq_dconly[:, 1:, :].abs().max()
    )

    print("original max |DC token|:", original_dc_magnitude)
    print()
    print("[No-DC]")
    print("max |DC token|:", nodc_dc)
    print("max |non-DC coefficient change|:", nodc_non_dc_change)
    print()
    print("[DC-only]")
    print("max |DC coefficient change|:", dconly_dc_change)
    print("max |non-DC token|:", dconly_non_dc)

    if original_dc_magnitude <= 1e-6:
        raise RuntimeError(
            "Sanity input unexpectedly has near-zero original DC."
        )

    tol = 1e-7

    if nodc_dc > tol:
        raise RuntimeError(
            "No-DC control did not fully remove k=0."
        )
    if nodc_non_dc_change > tol:
        raise RuntimeError(
            "No-DC control changed k>0 coefficients."
        )
    if dconly_dc_change > tol:
        raise RuntimeError(
            "DC-only control changed k=0."
        )
    if dconly_non_dc > tol:
        raise RuntimeError(
            "DC-only control did not fully remove k>0."
        )

    print()
    print(
        "PASS: No-DC changes only k=0; "
        "DC-only keeps only k=0."
    )


if __name__ == "__main__":
    main()
