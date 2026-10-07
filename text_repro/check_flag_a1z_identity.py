import math

import torch
import torch.nn.functional as F

from AMPCliff.factory.pooling.flag_pooling import (
    FFTLatentAttentionGatePooling,
)
from AMPCliff.factory.pooling.llm_pooling_dropin import (
    masked_mean_pooling,
)


def main():
    torch.manual_seed(0)

    batch = 4
    seq_len = 17
    d_model = 32

    x = torch.randn(batch, seq_len, d_model)
    mask = torch.zeros(batch, seq_len, dtype=torch.long)
    lengths = [17, 13, 9, 5]
    for i, length in enumerate(lengths):
        mask[i, :length] = 1

    pool = FFTLatentAttentionGatePooling(
        d_model=d_model,
        num_latents=8,
        num_heads=4,
        dropout=0.0,
        time_pool="mean",
        gate_residual=True,
        eps=1e-6,
        use_gate=True,
        use_latent=True,
        post_pool_norm=False,
        window_type=None,
        fixed_fft_length=None,
        remove_dc=False,
        dc_only=False,
        identity_time_out_proj=True,
        zero_init_gate_output=True,
    )
    pool.eval()

    with torch.no_grad():
        expected = masked_mean_pooling(x, mask, eps=1e-6)
        actual = pool(x, attention_mask=mask)

    diff = actual - expected
    max_abs = float(diff.abs().max())

    cos = F.cosine_similarity(actual, expected, dim=-1)
    cos = cos.clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cos))

    print("A1Z INITIAL-IDENTITY CHECK")
    print(f"max_abs_error={max_abs:.10e}")
    print(f"min_cosine={float(cos.min()):.10f}")
    print(f"max_angle_deg={float(angle.max()):.10f}")

    # FFT/iFFT roundoff is expected. This threshold is intentionally loose
    # enough for CPU/GPU float32 while still catching any real architectural
    # mismatch such as post-pool LayerNorm or a non-identity projection.
    if max_abs > 2e-5:
        raise SystemExit(
            "FAIL: A1Z + mean + no-post-norm is not Mean-equivalent "
            f"at initialization (max_abs_error={max_abs:.3e})."
        )

    print("PASS: A1Z starts from masked Mean up to FFT float32 roundoff.")


if __name__ == "__main__":
    main()
