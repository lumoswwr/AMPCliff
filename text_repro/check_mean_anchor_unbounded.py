#!/usr/bin/env python3
"""Sanity checks for unbounded Mean-anchor FLaG."""

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

from AMPCliff.factory.pooling.flag_pooling import FFTLatentAttentionGatePooling
from AMPCliff.factory.pooling.llm_pooling_dropin import masked_mean_pooling

def pair_cos(z):
    b = z.size(0) // 2
    return F.cosine_similarity(z[:b], z[b:], dim=-1)

def main():
    torch.manual_seed(23)

    x = torch.randn(8, 13, 16)
    lengths = torch.tensor([13,11,9,7,12,10,8,6])
    pos = torch.arange(x.size(1))[None,:]
    mask = (pos < lengths[:,None]).long()

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
        mean_anchor_unbounded=True,
    ).eval()

    if hasattr(model, "mean_anchor_beta_raw"):
        raise RuntimeError("Unbounded model unexpectedly created tanh beta.")
    if not hasattr(model, "mean_anchor_beta_unbounded"):
        raise RuntimeError("Missing direct unbounded beta parameter.")

    with torch.no_grad():
        mean = masked_mean_pooling(x, mask, eps=1e-6)

        model.mean_anchor_beta_unbounded.fill_(0.0)
        z0 = model(x, mask)

        model.mean_anchor_beta_unbounded.fill_(2.0)
        z2 = model(x, mask)

    endpoint_diff = float(
        (pair_cos(z0) - pair_cos(mean)).abs().max()
    )
    movement = float(
        (pair_cos(z2) - pair_cos(z0)).abs().max()
    )

    print("beta=0 max cosine diff vs Mean:", endpoint_diff)
    print("beta=2 max cosine change vs beta=0:", movement)
    print(
        "stored beta parameter:",
        float(model.mean_anchor_beta_unbounded.detach()),
    )

    if endpoint_diff > 2e-6:
        raise RuntimeError("beta=0 is not the exact Mean cosine endpoint.")
    if not (movement > 1e-6):
        raise RuntimeError("beta>1 does not affect the representation.")

    print("PASS: exact Mean endpoint and no |beta|<1 ceiling.")

if __name__ == "__main__":
    main()
