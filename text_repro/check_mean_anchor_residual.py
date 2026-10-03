#!/usr/bin/env python3
"""Sanity checks for the Mean-anchored orthogonal residual FLaG."""

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
    torch.manual_seed(11)

    x = torch.randn(8, 15, 16)
    lengths = torch.tensor([15, 13, 11, 9, 14, 12, 10, 8])
    pos = torch.arange(x.size(1))[None, :]
    mask = (pos < lengths[:, None]).long()

    model = FFTLatentAttentionGatePooling(
        d_model=16,
        num_latents=4,
        num_heads=4,
        dropout=0.0,
        time_pool="max",
        gate_residual=True,
        post_pool_norm=True,
        mean_anchor_residual=True,
        mean_anchor_beta_init=0.1,
    ).eval()

    with torch.no_grad():
        # beta=0 must be the exact Mean endpoint under cosine scoring.
        # A very negative logit approximates beta=0 closely enough to
        # verify the Mean endpoint numerically under cosine scoring.
        model.mean_anchor_beta_logit.fill_(-30.0)
        z_beta0 = model(x, mask)

        z_mean = masked_mean_pooling(
            x,
            mask,
            eps=1e-6,
        )

        mean_cos_diff = float(
            (
                pair_cos(z_beta0)
                - pair_cos(z_mean)
            ).abs().max()
        )

        # Run with nonzero beta so the stored residual is populated.
        beta = 0.3
        model.mean_anchor_beta_logit.fill_(
            torch.log(torch.tensor(beta / (1.0 - beta)))
        )
        _ = model(x, mask)

        mean_branch = model._last_mean_branch
        residual = model._last_orthogonal_residual

        max_abs_dot = float(
            (
                mean_branch * residual
            ).sum(dim=-1).abs().max()
        )

    print(
        "beta=0 max cosine diff vs Mean :",
        mean_cos_diff,
    )
    print(
        "max |<Mean, residual>|          :",
        max_abs_dot,
    )

    tol = 2e-6

    if mean_cos_diff > tol:
        raise RuntimeError(
            "beta=0 does not reproduce Mean cosine predictions."
        )

    if max_abs_dot > tol:
        raise RuntimeError(
            "Residual is not orthogonal to the Mean direction."
        )

    print(
        "PASS: beta=0 is exact Mean under cosine, and the "
        "FLaG residual is orthogonal to Mean."
    )


if __name__ == "__main__":
    main()
