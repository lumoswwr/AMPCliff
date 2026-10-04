#!/usr/bin/env python3
"""Verify stop-gradient AlignmentAnchor keeps the same forward map."""

from pathlib import Path
import sys
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = _REPO_ROOT.parent
for _p in (_REPO_PARENT, _REPO_ROOT / "text_repro"):
    _s=str(_p)
    if _s not in sys.path:
        sys.path.insert(0,_s)

from AMPCliff.factory.pooling.flag_pooling import FFTLatentAttentionGatePooling

def make(stopgrad):
    return FFTLatentAttentionGatePooling(
        d_model=16,
        num_latents=4,
        num_heads=4,
        dropout=0.0,
        time_pool="max",
        gate_residual=True,
        post_pool_norm=True,
        mean_anchor_residual=True,
        mean_anchor_beta_init=0.3,
        mean_alignment_anchor=True,
        mean_alignment_stopgrad=stopgrad,
    ).eval()

def main():
    torch.manual_seed(41)

    live=make(False)
    stop=make(True)
    stop.load_state_dict(live.state_dict(),strict=True)

    lengths=torch.tensor([13,11,9,7])
    pos=torch.arange(13)[None,:]
    mask=(pos<lengths[:,None]).long()

    x1=torch.randn(4,13,16,requires_grad=True)
    x2=x1.detach().clone().requires_grad_(True)

    y1=live(x1,mask)
    y2=stop(x2,mask)

    forward_diff=float((y1-y2).abs().max())

    probe=torch.randn_like(y1)
    (y1*probe).sum().backward()
    (y2*probe).sum().backward()

    grad_diff=float((x1.grad-x2.grad).abs().max())

    print("max forward diff:",forward_diff)
    print("max input-gradient diff:",grad_diff)

    if forward_diff>2e-6:
        raise RuntimeError("StopGrad changed the forward function.")
    if not (grad_diff>1e-8):
        raise RuntimeError("StopGrad did not change the backward path.")

    print("PASS: identical forward values, different alignment gradient path.")

if __name__=="__main__":
    main()
