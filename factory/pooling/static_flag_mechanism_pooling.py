"""Minimal trainable controls for mechanism tests against FLaG-B2.

All three controls start *exactly* as masked Mean:
  static_reim:   shared real/imag frequency-channel gate, with exact
                 FFT-free circular-reversal time-domain equivalent.
  static_diag:   one learned channel gate directly on masked Mean.
  mean_project:  masked Mean followed by an identity-initialized
                 TRAINABLE D-by-D output projection.

Gate parameterization follows FLaG-B2: 2 * sigmoid(logits), logits start 0.
Projection starts identity but remains trainable, matching FLaG-B2.
No FFT, latent attention, or gate MLP is needed for these controls.
"""
import torch
import torch.nn as nn


class StaticFLaGMechanismPooling(nn.Module):
    def __init__(self, d_model: int, mode: str):
        super().__init__()
        if mode not in {"static_reim", "static_diag", "mean_project", "mean_project_random"}:
            raise ValueError(f"Unsupported static control {mode!r}")
        self.d_model = int(d_model)
        self.mode = mode

        if mode == "static_reim":
            self.gate_logits = nn.Parameter(torch.zeros(2 * d_model))
        elif mode == "static_diag":
            self.gate_logits = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter("gate_logits", None)

        self.time_out_proj = nn.Linear(d_model, d_model)
        if mode != "mean_project_random":
            nn.init.eye_(self.time_out_proj.weight)
            nn.init.zeros_(self.time_out_proj.bias)

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor):
        if features.ndim != 3 or features.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected features [B,T,{self.d_model}], got {tuple(features.shape)}"
            )
        if attention_mask is None or attention_mask.shape != features.shape[:2]:
            raise ValueError("Expected attention_mask [B,T].")

        mask = attention_mask.unsqueeze(-1).to(features.dtype)
        x = features * mask
        denom = mask.sum(dim=1).clamp_min(1e-6)
        mean = x.sum(dim=1) / denom

        if self.mode in {"mean_project", "mean_project_random"}:
            pooled = mean
        else:
            g = 2.0 * torch.sigmoid(self.gate_logits)
            if self.mode == "static_diag":
                pooled = mean * g
            else:
                gr, gi = g.chunk(2, dim=-1)
                n = features.size(1)
                reverse_indices = (-torch.arange(n, device=features.device)) % n
                reversed_mean = (
                    x.index_select(1, reverse_indices) * mask
                ).sum(dim=1) / denom
                pooled = .5 * (gr + gi) * mean + .5 * (gr - gi) * reversed_mean

        return self.time_out_proj(pooled)
