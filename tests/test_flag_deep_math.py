"""Pure-torch checks for the channel-shared real/imag FLaG identity.

No Hugging Face models, data downloads, or CUDA required.
"""
import math

import torch


def masked_mean(x, mask):
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(1) / w.sum(1).clamp_min(1)


def time_closed_form(x, mask, gr, gi):
    n = x.shape[1]
    masked = x * mask.unsqueeze(-1)
    rev = masked.index_select(1, (-torch.arange(n)) % n)
    original = masked_mean(masked, mask)
    reverse = masked_mean(rev, mask)
    return .5 * (gr + gi) * original + .5 * (gr - gi) * reverse


def torch_fft_time_mean(x, mask, gr, gi):
    n = x.size(1)
    masked = x * mask.unsqueeze(-1)
    spec = torch.fft.rfft(masked, dim=1)
    altered = torch.complex(
        spec.real * gr.unsqueeze(1),
        spec.imag * gi.unsqueeze(1),
    )
    tokens = torch.fft.irfft(altered, n=n, dim=1)
    return masked_mean(tokens, mask)


def test_closed_form_matches_fft_even_odd_and_padding():
    torch.manual_seed(123)
    for n in (5, 6, 7, 8, 16, 33, 64):
        for d in (4, 32):
            x = torch.randn(4, n, d)
            mask = torch.zeros(4, n)
            lengths = [1, max(1, n//3), max(1, n//2), n]
            for row, length in enumerate(lengths):
                mask[row, :length] = 1
            gr = 2*torch.rand(4, d)
            gi = 2*torch.rand(4, d)
            a = time_closed_form(x, mask, gr, gi)
            b = torch_fft_time_mean(x, mask, gr, gi)
            assert torch.allclose(a, b, atol=2e-5, rtol=2e-5), (n, d)


def test_equal_real_imag_is_exact_channel_rescaling():
    torch.manual_seed(11)
    x = torch.randn(4, 13, 9)
    mask = torch.zeros(4, 13)
    mask[:, :7] = 1
    g = torch.rand(4, 9) + .5
    original = masked_mean(x, mask)
    actual = torch_fft_time_mean(x, mask, g, g)
    assert torch.allclose(actual, original*g, atol=2e-6, rtol=2e-6)


def test_unpadded_mean_depends_only_on_real_gate():
    torch.manual_seed(34)
    x = torch.randn(4, 9, 11)
    mask = torch.ones(4, 9)
    gr = torch.rand(4, 11)
    gi = torch.rand(4, 11)
    a = torch_fft_time_mean(x, mask, gr, gi)
    assert torch.allclose(a, gr*x.mean(1), atol=1e-6, rtol=1e-6)


def test_long_right_pad_reversed_mean_is_first_token_over_length():
    torch.manual_seed(2026)
    n, length, d = 30, 8, 6
    x = torch.randn(3, n, d)
    mask = torch.zeros(3, n)
    mask[:, :length] = 1
    gr, gi = torch.rand(3, d), torch.rand(3, d)
    mean = masked_mean(x, mask)
    predicted = (
        .5*(gr+gi)*mean + .5*(gr-gi)*x[:, 0, :]/length
    )
    fft = torch_fft_time_mean(x, mask, gr, gi)
    assert torch.allclose(fft, predicted, atol=2e-6, rtol=2e-6)


def test_dc_weighted_angle_exact():
    torch.manual_seed(42)
    x = torch.randn(13, 24)
    g = torch.rand(13, 24)*1.9 + .05
    w = x.square()
    w /= w.sum(-1, keepdim=True)
    e1 = (w*g).sum(-1)
    e2 = (w*g.square()).sum(-1)
    theoretical = e1/e2.sqrt()
    empirical = torch.nn.functional.cosine_similarity(x, x*g, dim=-1)
    assert torch.allclose(empirical, theoretical, atol=2e-6, rtol=2e-6)


def test_original_residual_gate_max_angle():
    # Kantorovich bound with positive scalings in [1, 2]:
    # cos(theta) >= 2*sqrt(1*2)/(1+2)
    expected_max = math.degrees(math.acos(2*math.sqrt(2)/3))
    assert abs(expected_max - 19.47122063449) < 1e-8

if __name__ == "__main__":
    test_closed_form_matches_fft_even_odd_and_padding()
    test_equal_real_imag_is_exact_channel_rescaling()
    test_unpadded_mean_depends_only_on_real_gate()
    test_long_right_pad_reversed_mean_is_first_token_over_length()
    test_dc_weighted_angle_exact()
    test_original_residual_gate_max_angle()
    print("PASS: 6 FLaG math identities / checks")
