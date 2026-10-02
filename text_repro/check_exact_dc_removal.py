#!/usr/bin/env python3
"""Numerically verify that FLaG DC removal zeros only k=0."""

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


def main():
    torch.manual_seed(0)

    # Variable valid lengths inside one dynamically padded batch.
    x = torch.randn(4, 11, 8)
    lengths = torch.tensor([11, 8, 5, 2])
    positions = torch.arange(x.size(1))[None, :]
    mask = (positions < lengths[:, None]).long()

    base = FFTLatentAttentionGatePooling(
        d_model=8,
        num_latents=2,
        num_heads=2,
        dropout=0.0,
        remove_dc=False,
    )
    nodc = FFTLatentAttentionGatePooling(
        d_model=8,
        num_latents=2,
        num_heads=2,
        dropout=0.0,
        remove_dc=True,
    )

    # We only test the analysis FFT, so learned parameter values are irrelevant.
    freq_base = base._to_frequency_tokens(
        x,
        attention_mask=mask,
    )
    freq_nodc = nodc._to_frequency_tokens(
        x,
        attention_mask=mask,
    )

    d = x.size(-1)

    # k=0 real and imaginary components must both be exactly zero.
    dc_real = freq_nodc[:, 0, :d]
    dc_imag = freq_nodc[:, 0, d:]
    max_abs_dc = max(
        float(dc_real.abs().max()),
        float(dc_imag.abs().max()),
    )

    # Every k>0 coefficient must be unchanged.
    max_abs_non_dc_change = float(
        (
            freq_nodc[:, 1:, :]
            - freq_base[:, 1:, :]
        ).abs().max()
    )

    # Sanity: the original DC should actually be nonzero.
    original_dc_magnitude = float(
        freq_base[:, 0, :].abs().max()
    )

    print("original max |DC token|:", original_dc_magnitude)
    print("after ablation max |DC token|:", max_abs_dc)
    print("max |non-DC coefficient change|:", max_abs_non_dc_change)

    if original_dc_magnitude <= 1e-6:
        raise RuntimeError("Sanity input unexpectedly has near-zero original DC.")
    if max_abs_dc > 1e-7:
        raise RuntimeError("DC coefficient was not fully removed.")
    if max_abs_non_dc_change > 1e-7:
        raise RuntimeError(
            "Non-DC coefficients changed; ablation is not spectrally exact."
        )

    print("PASS: k=0 is zero and every k>0 coefficient is unchanged.")


if __name__ == "__main__":
    main()
