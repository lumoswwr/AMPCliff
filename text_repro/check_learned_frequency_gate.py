#!/usr/bin/env python3
from pathlib import Path
import sys
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_PARENT = _REPO_ROOT.parent
for _p in (_REPO_PARENT, _REPO_ROOT / "text_repro"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from AMPCliff.factory.pooling.flag_pooling import FFTLatentAttentionGatePooling

def main():
    torch.manual_seed(7)
    B,K,D,M = 2,9,8,4
    freq_tokens = torch.randn(B,K,2*D)
    latent_out = torch.randn(B,M,2*D)

    base = FFTLatentAttentionGatePooling(
        d_model=D, num_latents=M, num_heads=4, dropout=0.0,
        learned_frequency_gate=False,
    ).eval()
    learned = FFTLatentAttentionGatePooling(
        d_model=D, num_latents=M, num_heads=4, dropout=0.0,
        learned_frequency_gate=True, learned_frequency_hidden=16,
    ).eval()

    # Copy only shared parameters; learned scorer has extra identity-init params.
    shared = learned.state_dict()
    for k,v in base.state_dict().items():
        if k in shared and shared[k].shape == v.shape:
            shared[k] = v
    learned.load_state_dict(shared, strict=True)

    out_base = base._apply_gate(freq_tokens, latent_out)
    out_learned = learned._apply_gate(freq_tokens, latent_out)

    max_diff = float((out_base - out_learned).abs().max())
    gate = learned._last_frequency_gate
    gate_diff = float((gate - 1.0).abs().max())

    loss = out_learned.square().mean()
    loss.backward()
    grad_norm = float(learned.frequency_scorer_out.weight.grad.abs().sum())

    print("identity max output diff vs original FLaG:", max_diff)
    print("identity max |frequency_gate - 1|:", gate_diff)
    print("scorer output-layer grad L1:", grad_norm)

    tol = 2e-6
    if max_diff > tol:
        raise RuntimeError("Learned gate does not start from original FLaG.")
    if gate_diff > tol:
        raise RuntimeError("Learned gate is not identity-initialized.")
    if not (grad_norm > 0.0):
        raise RuntimeError("Learned frequency scorer receives no gradient.")

    print("PASS: exact identity initialization with live gradient.")

if __name__ == "__main__":
    main()
