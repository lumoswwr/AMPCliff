#!/usr/bin/env python3
"""Verify the exact Mean/FLaG endpoints of FLaG_MeanResidual."""

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = _REPO_ROOT.parent
for _p in (_REPO_PARENT, _REPO_ROOT / "text_repro"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from AMPCliff.factory.pooling.flag_pooling import (
    FFTLatentAttentionGatePooling,
)
from AMPCliff.factory.pooling.llm_pooling_dropin import (
    masked_mean_pooling,
)


def pair_cos(z):
    b = z.size(0) // 2
    return F.cosine_similarity(
        z[:b],
        z[b:],
        dim=-1,
    )


def main():
    torch.manual_seed(7)

    x = torch.randn(8, 13, 16)
    lengths = torch.tensor([13, 11, 9, 7, 12, 10, 8, 6])
    pos = torch.arange(x.size(1))[None, :]
    mask = (pos < lengths[:, None]).long()

    base = FFTLatentAttentionGatePooling(
        d_model=16,
        num_latents=4,
        num_heads=4,
        dropout=0.0,
        time_pool="max",
        gate_residual=True,
        post_pool_norm=True,
        mean_residual=False,
    ).eval()

    mixed = FFTLatentAttentionGatePooling(
        d_model=16,
        num_latents=4,
        num_heads=4,
        dropout=0.0,
        time_pool="max",
        gate_residual=True,
        post_pool_norm=True,
        mean_residual=True,
        mean_mix_init=0.5,
    ).eval()

    missing, unexpected = mixed.load_state_dict(
        base.state_dict(),
        strict=False,
    )

    if missing != ["mean_mix_alpha"] or unexpected:
        raise RuntimeError(
            f"Unexpected state mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )

    with torch.no_grad():
        z_base = base(x, mask)
        z_mean = masked_mean_pooling(
            x,
            mask,
            eps=1e-6,
        )

        mixed.mean_mix_alpha.fill_(0.0)
        z_alpha0 = mixed(x, mask)

        mixed.mean_mix_alpha.fill_(1.0)
        z_alpha1 = mixed(x, mask)

    mean_cos_diff = float(
        (
            pair_cos(z_alpha0)
            - pair_cos(z_mean)
        ).abs().max()
    )
    flag_cos_diff = float(
        (
            pair_cos(z_alpha1)
            - pair_cos(z_base)
        ).abs().max()
    )

    print("alpha=0 max cosine diff vs Mean :", mean_cos_diff)
    print("alpha=1 max cosine diff vs FLaG :", flag_cos_diff)

    tol = 2e-6
    if mean_cos_diff > tol:
        raise RuntimeError(
            "alpha=0 does not reproduce Mean cosine predictions."
        )
    if flag_cos_diff > tol:
        raise RuntimeError(
            "alpha=1 does not reproduce original FLaG cosine predictions."
        )

    print(
        "PASS: alpha=0 is the Mean endpoint and "
        "alpha=1 is the original FLaG endpoint under cosine scoring."
    )


if __name__ == "__main__":
    main()
