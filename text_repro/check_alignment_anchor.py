#!/usr/bin/env python3
"""Sanity checks for alignment-aware Mean anchor."""

from pathlib import Path
import sys
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = _REPO_ROOT.parent
for _p in (_REPO_PARENT, _REPO_ROOT / "text_repro"):
    _s=str(_p)
    if _s not in sys.path:
        sys.path.insert(0,_s)

from AMPCliff.factory.pooling.flag_pooling import FFTLatentAttentionGatePooling
from AMPCliff.factory.pooling.llm_pooling_dropin import masked_mean_pooling

def pair_cos(z):
    b=z.size(0)//2
    return F.cosine_similarity(z[:b],z[b:],dim=-1)

def main():
    torch.manual_seed(31)

    x=torch.randn(8,13,16)
    lengths=torch.tensor([13,11,9,7,12,10,8,6])
    pos=torch.arange(x.size(1))[None,:]
    mask=(pos<lengths[:,None]).long()

    base=FFTLatentAttentionGatePooling(
        d_model=16,num_latents=4,num_heads=4,dropout=0.0,
        time_pool="max",gate_residual=True,post_pool_norm=True,
    ).eval()

    aa=FFTLatentAttentionGatePooling(
        d_model=16,num_latents=4,num_heads=4,dropout=0.0,
        time_pool="max",gate_residual=True,post_pool_norm=True,
        mean_anchor_residual=True,
        mean_anchor_beta_init=0.1,
        mean_alignment_anchor=True,
    ).eval()

    missing,unexpected=aa.load_state_dict(base.state_dict(),strict=False)
    if missing != ["mean_anchor_beta_unbounded"] or unexpected:
        raise RuntimeError(
            f"Unexpected state mismatch: missing={missing}, unexpected={unexpected}"
        )

    with torch.no_grad():
        zf=base(x,mask)
        zm=masked_mean_pooling(x,mask,eps=1e-6)

        aa.mean_anchor_beta_unbounded.fill_(0.0)
        z0=aa(x,mask)

        aa.mean_anchor_beta_unbounded.fill_(1.0)
        z1=aa(x,mask)

        m=F.normalize(zm,p=2,dim=-1)
        f=F.normalize(zf,p=2,dim=-1)
        target=F.normalize(m+f,p=2,dim=-1)

    mean_diff=float((pair_cos(z0)-pair_cos(zm)).abs().max())
    sum_diff=float((pair_cos(z1)-pair_cos(target)).abs().max())

    print("beta=0 max cosine diff vs Mean:",mean_diff)
    print("beta=1 max cosine diff vs normalize(Mean+FLaG):",sum_diff)

    if mean_diff>2e-6:
        raise RuntimeError("beta=0 is not Mean-equivalent.")
    if sum_diff>2e-6:
        raise RuntimeError("beta=1 is not normalize(Mean+FLaG)-equivalent.")

    print("PASS: alignment-aware anchor endpoints are correct.")

if __name__=="__main__":
    main()
