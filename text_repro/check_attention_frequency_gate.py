#!/usr/bin/env python3
"""Sanity checks for the attention-derived frequency-wise FLaG gate."""

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
    torch.manual_seed(13)

    B = 3
    K = 9
    D = 8
    M = 4

    freq_tokens = torch.randn(B, K, 2 * D)
    latent_out = torch.randn(B, M, 2 * D)

    base = FFTLatentAttentionGatePooling(
        d_model=D,
        num_latents=M,
        num_heads=4,
        dropout=0.0,
        attention_frequency_gate=False,
    ).eval()

    gated = FFTLatentAttentionGatePooling(
        d_model=D,
        num_latents=M,
        num_heads=4,
        dropout=0.0,
        attention_frequency_gate=True,
    ).eval()

    gated.load_state_dict(
        base.state_dict(),
        strict=True,
    )

    uniform = torch.full(
        (B, M, K),
        1.0 / K,
        dtype=freq_tokens.dtype,
    )

    gated._current_latent_attn_weights = uniform
    out_gated = gated._apply_gate(
        freq_tokens,
        latent_out,
    )
    out_base = base._apply_gate(
        freq_tokens,
        latent_out,
    )

    max_identity_diff = float(
        (out_gated - out_base).abs().max()
    )

    gate_uniform = gated._last_frequency_gate
    max_uniform_gate_diff = float(
        (gate_uniform - 1.0).abs().max()
    )

    # Concentrate all attention on DC. The proposed map should then produce
    # a high DC gate and exact zero gates for all non-DC bins.
    dc_only = torch.zeros(
        B,
        M,
        K,
        dtype=freq_tokens.dtype,
    )
    dc_only[:, :, 0] = 1.0

    gated._current_latent_attn_weights = dc_only
    _ = gated._apply_gate(
        freq_tokens,
        latent_out,
    )

    freq_gate = gated._last_frequency_gate
    dc_gate = float(freq_gate[:, 0].mean())
    non_dc_max = float(
        freq_gate[:, 1:].abs().max()
    )

    print(
        "uniform attention max output diff vs original FLaG:",
        max_identity_diff,
    )
    print(
        "uniform attention max |frequency_gate - 1|:",
        max_uniform_gate_diff,
    )
    print("DC-focused attention mean g0:", dc_gate)
    print("DC-focused attention max non-DC gate:", non_dc_max)

    tol = 2e-6

    if max_identity_diff > tol:
        raise RuntimeError(
            "Uniform attention does not reproduce original FLaG gating."
        )

    if max_uniform_gate_diff > tol:
        raise RuntimeError(
            "Uniform attention does not map to unit frequency gates."
        )

    if not (1.0 < dc_gate < 2.0):
        raise RuntimeError(
            f"Expected enhanced DC gate in (1,2), got {dc_gate}"
        )

    if non_dc_max > tol:
        raise RuntimeError(
            "Zero-attention non-DC bins were not fully suppressed."
        )

    print(
        "PASS: uniform attention is identity; concentrated attention "
        "produces true frequency-wise suppression/enhancement."
    )


if __name__ == "__main__":
    main()
