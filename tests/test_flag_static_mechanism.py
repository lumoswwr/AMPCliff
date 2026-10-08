"""CPU-only forward and gradient checks for static FLaG mechanism controls."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factory.pooling.static_flag_mechanism_pooling import StaticFLaGMechanismPooling


def mean(x, mask):
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x*w).sum(1) / w.sum(1).clamp_min(1)


def test_identity_at_epoch0():
    torch.manual_seed(31)
    x = torch.randn(4, 17, 32)
    mask = torch.zeros(4, 17)
    mask[0,:17],mask[1,:13],mask[2,:8],mask[3,:2] = 1,1,1,1
    expected = mean(x, mask)
    for mode in ("mean_project", "static_diag", "static_reim"):
        pool = StaticFLaGMechanismPooling(32, mode)
        got = pool(x, mask)
        assert torch.allclose(got, expected, atol=1e-6, rtol=1e-6), mode


def test_random_projection_initialization():
    torch.manual_seed(111)
    p = StaticFLaGMechanismPooling(32, "mean_project_random")
    ident = torch.eye(32)
    assert not torch.allclose(p.time_out_proj.weight, ident, atol=1e-4)


def test_static_reim_exact_fft_equivalence_and_gradients():
    torch.manual_seed(5)
    n, d = 29, 16
    x = torch.randn(3,n,d,requires_grad=True)
    mask = torch.zeros(3,n)
    mask[0,:5],mask[1,:17],mask[2,:29] = 1,1,1
    pool = StaticFLaGMechanismPooling(d, "static_reim")
    with torch.no_grad():
        pool.gate_logits.copy_(torch.randn(2*d))
    got = pool(x,mask)
    gr,gi = (2*pool.gate_logits.sigmoid()).chunk(2)
    freq = torch.fft.rfft(x*mask.unsqueeze(-1),dim=1)
    gated = torch.complex(freq.real*gr, freq.imag*gi)
    inverse = torch.fft.irfft(gated,n=n,dim=1)
    expected = pool.time_out_proj(mean(inverse,mask))
    assert torch.allclose(got,expected,atol=5e-6,rtol=5e-6)
    got.square().sum().backward()
    assert x.grad is not None and x.grad.abs().sum()>0
    assert pool.gate_logits.grad is not None
    assert pool.gate_logits.grad.abs().sum()>0
    assert pool.time_out_proj.weight.grad is not None


def test_static_diag_equals_equal_reim():
    torch.manual_seed(37)
    x = torch.randn(2,14,12)
    mask = torch.zeros(2,14)
    mask[0,:9],mask[1,:14] = 1,1
    one = StaticFLaGMechanismPooling(12,"static_diag")
    two = StaticFLaGMechanismPooling(12,"static_reim")
    with torch.no_grad():
        p = torch.randn(12)
        one.gate_logits.copy_(p)
        two.gate_logits.copy_(torch.cat([p,p]))
    assert torch.allclose(one(x,mask),two(x,mask),atol=1e-6,rtol=1e-6)


if __name__=="__main__":
    test_identity_at_epoch0()
    test_random_projection_initialization()
    test_static_reim_exact_fft_equivalence_and_gradients()
    test_static_diag_equals_equal_reim()
    print("PASS: 4 static-FLaG controls (identity, FFT equivalence, gradients)")
