"""FLaG core pooling: FFT latent attention gate (release-only module)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llm_pooling_dropin import masked_max_pooling, masked_mean_pooling


class FFTLatentAttentionGatePooling(nn.Module):
    """
    FFT latent-attention pooling with gate-based frequency modulation.

    Pipeline:
        optional windowing
        -> rFFT
        -> latent attention
        -> gate
        -> iFFT
        -> time pooling
        -> projection
    """

    def __init__(
        self,
        d_model: int,
        num_latents: int = 16,
        num_heads: int = 8,
        dropout: float = 0.1,
        time_pool: str = "max",
        gate_residual: bool = True,
        eps: float = 1e-6,
        use_gate: bool = True,
        use_latent: bool = True,
        post_pool_norm: bool = False,
        window_type: Optional[str] = None,
        fixed_fft_length: Optional[int] = None,
        remove_dc: bool = False,
        dc_only: bool = False,
        mean_residual: bool = False,
        mean_mix_init: float = 0.5,
        mean_anchor_residual: bool = False,
        mean_anchor_beta_init: float = 0.1,
        mean_anchor_unbounded: bool = False,
        mean_alignment_anchor: bool = False,
        mean_alignment_stopgrad: bool = False,
        attention_frequency_gate: bool = False,
        learned_frequency_gate: bool = False,
        learned_frequency_hidden: int = 256,
        identity_time_out_proj: bool = False,
        zero_init_gate_output: bool = False,
    ):
        super().__init__()

        freq_dim = d_model * 2

        if freq_dim % num_heads != 0:
            raise ValueError(
                f"2*d_model={freq_dim} must be divisible by num_heads={num_heads}"
            )

        if time_pool not in {"mean", "max"}:
            raise ValueError(
                f"time_pool must be 'mean' or 'max', got {time_pool}"
            )

        if window_type not in {None, "hann"}:
            raise ValueError(
                f"window_type must be None or 'hann', got {window_type}"
            )

        self.d_model = d_model
        self.freq_dim = freq_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.time_pool = time_pool
        self.gate_residual = bool(gate_residual)
        self.use_gate = bool(use_gate)
        self.use_latent = bool(use_latent)
        self.post_pool_norm = bool(post_pool_norm)
        self.window_type = window_type
        self.remove_dc = bool(remove_dc)
        self.dc_only = bool(dc_only)
        self.mean_residual = bool(mean_residual)
        self.mean_anchor_residual = bool(mean_anchor_residual)
        self.mean_anchor_unbounded = bool(mean_anchor_unbounded)
        self.mean_alignment_anchor = bool(mean_alignment_anchor)
        self.mean_alignment_stopgrad = bool(mean_alignment_stopgrad)
        self.attention_frequency_gate = bool(attention_frequency_gate)
        self.learned_frequency_gate = bool(learned_frequency_gate)
        self.learned_frequency_hidden = int(learned_frequency_hidden)
        self.identity_time_out_proj = bool(identity_time_out_proj)
        self.zero_init_gate_output = bool(zero_init_gate_output)

        if self.zero_init_gate_output and not self.use_gate:
            raise ValueError(
                "zero_init_gate_output=True requires use_gate=True."
            )

        if self.zero_init_gate_output and not self.gate_residual:
            raise ValueError(
                "zero_init_gate_output=True requires gate_residual=True."
            )

        if self.attention_frequency_gate and not self.use_latent:
            raise ValueError(
                "attention_frequency_gate=True requires use_latent=True."
            )

        if self.learned_frequency_gate and not self.use_latent:
            raise ValueError(
                "learned_frequency_gate=True requires use_latent=True."
            )

        if self.attention_frequency_gate and self.learned_frequency_gate:
            raise ValueError(
                "attention_frequency_gate and learned_frequency_gate "
                "are mutually exclusive."
            )

        if self.learned_frequency_hidden <= 0:
            raise ValueError(
                "learned_frequency_hidden must be > 0, "
                f"got {learned_frequency_hidden}"
            )

        if self.mean_residual and self.mean_anchor_residual:
            raise ValueError(
                "mean_residual and mean_anchor_residual are mutually exclusive."
            )

        if self.mean_anchor_unbounded and not self.mean_anchor_residual:
            raise ValueError(
                "mean_anchor_unbounded=True requires mean_anchor_residual=True."
            )

        if self.mean_alignment_anchor and not self.mean_anchor_residual:
            raise ValueError(
                "mean_alignment_anchor=True requires mean_anchor_residual=True."
            )

        if self.mean_alignment_stopgrad and not self.mean_alignment_anchor:
            raise ValueError(
                "mean_alignment_stopgrad=True requires mean_alignment_anchor=True."
            )

        if not (0.0 <= float(mean_mix_init) <= 1.0):
            raise ValueError(
                "mean_mix_init must be in [0, 1], "
                f"got {mean_mix_init}"
            )

        if self.mean_residual:
            # Direct scalar parameter rather than sigmoid(logit), so the
            # hypothesis space contains the exact endpoints alpha=0 (Mean)
            # and alpha=1 (original FLaG). The forward pass clamps alpha to
            # [0, 1]. Runners exclude this scalar from weight decay.
            self.mean_mix_alpha = nn.Parameter(
                torch.tensor(float(mean_mix_init))
            )

        if not (0.0 <= float(mean_anchor_beta_init) <= 1.0):
            raise ValueError(
                "mean_anchor_beta_init must be in [0, 1], "
                f"got {mean_anchor_beta_init}"
            )

        if self.mean_anchor_residual:
            # Mean-anchored complementary residual.
            #
            # Bounded version: beta=tanh(raw), so |beta|<1.
            # Unbounded control: beta is learned directly. This tests whether
            # Sprint's previous beta≈1 result was limited by tanh saturation.
            beta_init = float(mean_anchor_beta_init)

            if self.mean_anchor_unbounded or self.mean_alignment_anchor:
                self.mean_anchor_beta_unbounded = nn.Parameter(
                    torch.tensor(beta_init)
                )
            else:
                beta_init = min(
                    max(beta_init, -0.999999),
                    0.999999,
                )
                beta_raw_init = torch.atanh(
                    torch.tensor(beta_init)
                )
                self.mean_anchor_beta_raw = nn.Parameter(
                    beta_raw_init
                )

        if self.remove_dc and self.dc_only:
            raise ValueError(
                "remove_dc and dc_only are mutually exclusive."
            )

        self.eps = float(eps)

        if (
            fixed_fft_length is not None
            and fixed_fft_length <= 0
        ):
            raise ValueError(
                "fixed_fft_length must be > 0 or None, "
                f"got {fixed_fft_length}"
            )

        self.fixed_fft_length = (
            None
            if fixed_fft_length is None
            else int(fixed_fft_length)
        )

        if self.use_latent:
            self.latents = nn.Parameter(
                torch.randn(num_latents, freq_dim) * 0.02
            )

            self.attn = nn.MultiheadAttention(
                embed_dim=freq_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

            self.norm1 = nn.LayerNorm(freq_dim)

            self.ffn = nn.Sequential(
                nn.Linear(freq_dim, freq_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(freq_dim, freq_dim),
            )

            self.norm2 = nn.LayerNorm(freq_dim)

        if self.use_gate:
            self.freq_gate = nn.Sequential(
                nn.Linear(freq_dim, freq_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(freq_dim, freq_dim),
            )

            if self.zero_init_gate_output:
                # FLaG-zero / A1Z:
                # zero-initialize the final gate-output layer. In the
                # zero-init mode _apply_gate interprets this output as a
                # zero-centered residual delta via tanh, so raw=0 gives an
                # exact multiplicative identity of 1 at initialization.
                nn.init.zeros_(self.freq_gate[-1].weight)
                nn.init.zeros_(self.freq_gate[-1].bias)

        if self.learned_frequency_gate:
            # Proposal 2B: a frequency-specific scorer that does not reuse
            # latent-attention probabilities as the gate itself.
            #
            # For each frequency bin k:
            #   h_k = W_x X_k + W_s s + W_p p_k
            #   g_k = 2 * sigmoid(w^T GELU(LN(h_k)))
            #
            # where s is the sentence-level latent summary and p_k is the
            # normalized frequency coordinate in [0, 1]. The final scalar
            # projection is zero-initialized, so every g_k starts exactly at
            # 1 and the whole module initially reproduces original FLaG.
            hidden = self.learned_frequency_hidden

            self.frequency_token_proj = nn.Linear(
                freq_dim,
                hidden,
            )
            self.frequency_summary_proj = nn.Linear(
                freq_dim,
                hidden,
            )
            self.frequency_position_proj = nn.Linear(
                1,
                hidden,
                bias=False,
            )
            self.frequency_scorer_norm = nn.LayerNorm(
                hidden
            )
            self.frequency_scorer_dropout = nn.Dropout(
                dropout
            )
            self.frequency_scorer_out = nn.Linear(
                hidden,
                1,
            )

            nn.init.zeros_(
                self.frequency_scorer_out.weight
            )
            nn.init.zeros_(
                self.frequency_scorer_out.bias
            )

        self.time_out_proj = nn.Linear(d_model, d_model)

        if self.identity_time_out_proj:
            # FLaG-A1 / A1Z: start from an exact identity readout while
            # keeping the projection trainable after initialization.
            nn.init.eye_(self.time_out_proj.weight)
            nn.init.zeros_(self.time_out_proj.bias)

        self.dropout = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        if self.use_latent and not self.use_gate:
            raise ValueError(
                "use_latent=True requires use_gate=True; "
                "fft_latent_only ablation has been removed."
            )

    def _build_hann_window(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build one Hann window per sample using that sample's valid token length.

        Important:
        - Padding positions remain zero.
        - Different sequence lengths get different Hann windows.
        - Works even if valid positions are not strictly a prefix.
        """
        B, T, _ = features.shape

        if attention_mask is None:
            window = torch.hann_window(
                T,
                periodic=False,
                dtype=features.dtype,
                device=features.device,
            )
            return window.unsqueeze(0).expand(B, -1)

        mask = attention_mask.bool()

        windows = torch.zeros(
            B,
            T,
            dtype=features.dtype,
            device=features.device,
        )

        for b in range(B):
            valid_idx = torch.nonzero(
                mask[b],
                as_tuple=False,
            ).squeeze(-1)

            valid_len = valid_idx.numel()

            if valid_len == 0:
                continue

            # Extremely short sequences are pathological for a symmetric Hann:
            # length 1 or 2 can suppress essentially everything.
            if valid_len <= 2:
                w = torch.ones(
                    valid_len,
                    dtype=features.dtype,
                    device=features.device,
                )
            else:
                w = torch.hann_window(
                    valid_len,
                    periodic=False,
                    dtype=features.dtype,
                    device=features.device,
                )

            windows[b, valid_idx] = w

        return windows

    def _apply_input_window(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = features

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)

        if self.window_type is None:
            self._last_input_window = None
            return x

        if self.window_type == "hann":
            window = self._build_hann_window(
                features,
                attention_mask,
            )

            self._last_input_window = window.detach()

            return x * window.unsqueeze(-1)

        raise RuntimeError(
            f"Unsupported window_type: {self.window_type}"
        )

    def _to_frequency_tokens(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        x = self._apply_input_window(
            features,
            attention_mask,
        )

        fft_length = (
            self.fixed_fft_length
            if self.fixed_fft_length is not None
            else x.size(1)
        )

        self._last_fft_length = int(
            fft_length
        )

        spec = torch.fft.rfft(
            x,
            n=fft_length,
            dim=1,
        )

        # Exact spectral DC ablation.
        #
        # Important: do this AFTER the actual masked/windowed rFFT rather
        # than subtracting the valid-token mean in time space. With dynamic
        # right padding, valid-token centering would also perturb non-DC
        # coefficients of the batch-length FFT. Multiplying only k=0 by zero
        # leaves every non-DC coefficient bit-for-bit unchanged apart from
        # ordinary floating-point multiplication.
        if self.remove_dc or self.dc_only:
            freq_mask = torch.ones(
                spec.size(1),
                dtype=spec.real.dtype,
                device=spec.device,
            )

            if self.remove_dc:
                # Keep every non-DC frequency and remove only k=0.
                freq_mask[0] = 0.0
            else:
                # Keep exactly k=0 and remove every non-DC frequency.
                freq_mask[1:] = 0.0

            spec = (
                spec
                * freq_mask.view(1, -1, 1)
            )

        self._last_remove_dc = self.remove_dc
        self._last_dc_only = self.dc_only

        return torch.cat(
            [spec.real, spec.imag],
            dim=-1,
        )

    def _latent_pool_in_frequency(
        self,
        freq_tokens: torch.Tensor,
        freq_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = freq_tokens.size(0)

        queries = self.latents.unsqueeze(0).expand(
            B,
            -1,
            -1,
        )

        key_padding_mask = None

        if freq_attention_mask is not None:
            key_padding_mask = ~freq_attention_mask.bool()

        attn_out, attn_weights = self.attn(
            query=queries,
            key=freq_tokens,
            value=freq_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )

        if attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=1)

        latent_out = self.norm1(
            queries + self.dropout(attn_out)
        )

        latent_out = self.norm2(
            latent_out
            + self.dropout(
                self.ffn(latent_out)
            )
        )

        # Keep one live reference only until _apply_gate consumes it. This
        # allows the frequency gate to backpropagate through the same latent
        # attention weights without changing the public/private return type
        # of this helper.
        self._current_latent_attn_weights = attn_weights

        self._last_latent_attn_weights = (
            attn_weights.detach()
        )

        self._last_latent_out = (
            latent_out.detach()
        )

        self._last_latent_summary = (
            latent_out.mean(dim=1).detach()
        )

        return latent_out
        
    
    

    def _apply_gate(
        self,
        freq_tokens: torch.Tensor,
        latent_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if not self.use_gate:
            return freq_tokens

        if self.use_latent and latent_out is not None:
            gate_input = latent_out.mean(dim=1)
        else:
            gate_input = freq_tokens.mean(dim=1)

        gate_logits = self.freq_gate(gate_input)

        if self.zero_init_gate_output:
            # Identity-centered residual gate used only by FLaG-zero/A1Z:
            #
            #   delta = tanh(logits)
            #   multiplier = 1 + delta
            #
            # The zero-initialized final linear layer therefore gives
            # multiplier == 1 exactly at initialization. This is deliberately
            # isolated behind a flag so the original FLaG parameterization
            # (1 + sigmoid(logits)) is unchanged for all existing experiments.
            gate_delta = torch.tanh(gate_logits)
            self._last_raw_gate = gate_delta.detach()
            gate = 1.0 + gate_delta
        else:
            gate = torch.sigmoid(gate_logits)
            self._last_raw_gate = gate.detach()

            if self.gate_residual:
                gate = 1.0 + gate

        enhanced_freq = (
            freq_tokens * gate.unsqueeze(1)
        )

        if self.attention_frequency_gate:
            if not hasattr(
                self,
                "_current_latent_attn_weights",
            ):
                raise RuntimeError(
                    "Attention-derived frequency gate requires "
                    "latent attention weights from the current forward pass."
                )

            # [B, M, K] -> [B, K]
            #
            # Average over latent queries, then renormalize because attention
            # dropout can make the returned training-time weights deviate
            # slightly from a unit sum.
            freq_importance = (
                self._current_latent_attn_weights
                .mean(dim=1)
                .clamp_min(0.0)
            )

            denom = (
                freq_importance
                .sum(dim=1, keepdim=True)
                .clamp_min(self.eps)
            )
            freq_importance = (
                freq_importance / denom
            )

            # Uniform attention should be the identity:
            #   a_k = 1/K -> r_k = 1 -> g_k = 1.
            #
            # The bounded map gives genuine suppression and enhancement:
            #   r=0      -> g=0
            #   r=1      -> g=1
            #   r->inf   -> g->2
            num_freqs = freq_tokens.size(1)
            relative_importance = (
                float(num_freqs)
                * freq_importance
            )
            frequency_gate = (
                2.0
                * relative_importance
                / (1.0 + relative_importance)
            )

            enhanced_freq = (
                enhanced_freq
                * frequency_gate.unsqueeze(-1)
            )

            self._last_frequency_attention_importance = (
                freq_importance.detach()
            )
            self._last_frequency_gate = (
                frequency_gate.detach()
            )

        if self.learned_frequency_gate:
            if latent_out is None:
                raise RuntimeError(
                    "Learned frequency gate requires latent_out."
                )

            B, K, _ = freq_tokens.shape

            latent_summary = latent_out.mean(
                dim=1
            )

            if K <= 1:
                frequency_position = torch.zeros(
                    1,
                    K,
                    1,
                    dtype=freq_tokens.dtype,
                    device=freq_tokens.device,
                )
            else:
                frequency_position = torch.linspace(
                    0.0,
                    1.0,
                    K,
                    dtype=freq_tokens.dtype,
                    device=freq_tokens.device,
                ).view(1, K, 1)

            scorer_hidden = (
                self.frequency_token_proj(
                    freq_tokens
                )
                + self.frequency_summary_proj(
                    latent_summary
                ).unsqueeze(1)
                + self.frequency_position_proj(
                    frequency_position
                )
            )

            scorer_hidden = (
                self.frequency_scorer_norm(
                    scorer_hidden
                )
            )
            scorer_hidden = F.gelu(
                scorer_hidden
            )
            scorer_hidden = (
                self.frequency_scorer_dropout(
                    scorer_hidden
                )
            )

            raw_frequency_score = (
                self.frequency_scorer_out(
                    scorer_hidden
                ).squeeze(-1)
            )

            # Identity-centered bounded gate:
            # raw=0 -> g_k=1 exactly;
            # raw->-inf -> 0; raw->+inf -> 2.
            frequency_gate = (
                2.0
                * torch.sigmoid(
                    raw_frequency_score
                )
            )

            enhanced_freq = (
                enhanced_freq
                * frequency_gate.unsqueeze(-1)
            )

            self._last_learned_frequency_score = (
                raw_frequency_score.detach()
            )
            self._last_frequency_gate = (
                frequency_gate.detach()
            )
            self._last_frequency_position = (
                frequency_position.detach()
            )

        # Do not retain the autograd graph through a module attribute after
        # the frequency gate has consumed the live attention weights.
        if hasattr(
            self,
            "_current_latent_attn_weights",
        ):
            del self._current_latent_attn_weights

        self._last_freq_tokens = (
            freq_tokens.detach()
        )

        self._last_enhanced_freq = (
            enhanced_freq.detach()
        )

        return enhanced_freq

    def _back_to_time(
        self,
        enhanced_freq: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:

        real, imag = enhanced_freq.chunk(
            2,
            dim=-1,
        )

        spec = torch.complex(
            real,
            imag,
        )

        reconstruction_len = (
            self.fixed_fft_length
            if self.fixed_fft_length is not None
            else seq_len
        )

        time_tokens = torch.fft.irfft(
            spec,
            n=reconstruction_len,
            dim=1,
        )

        if reconstruction_len != seq_len:
            time_tokens = time_tokens[
                :, :seq_len, :
            ]

        return time_tokens

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_pre_projection: bool = False,
    ):
        _, T, D = features.shape

        if D != self.d_model:
            raise ValueError(
                f"Expected last dim {self.d_model}, got {D}"
            )

        if (
            self.fixed_fft_length is not None
            and T > self.fixed_fft_length
        ):
            raise ValueError(
                f"Input sequence length T={T} exceeds "
                f"fixed_fft_length={self.fixed_fft_length}"
            )

        freq_tokens = self._to_frequency_tokens(
            features,
            attention_mask=attention_mask,
        )

        self._last_seq_len = T

        if self.use_latent:
            latent_out = self._latent_pool_in_frequency(
                freq_tokens
            )

            enhanced_freq = self._apply_gate(
                freq_tokens,
                latent_out,
            )

        else:
            enhanced_freq = self._apply_gate(
                freq_tokens,
                latent_out=None,
            )

        time_tokens = self._back_to_time(
            enhanced_freq,
            seq_len=T,
        )

        if self.time_pool == "mean":
            pooled = masked_mean_pooling(
                time_tokens,
                attention_mask,
                eps=self.eps,
            )

        else:
            pooled = masked_max_pooling(
                time_tokens,
                attention_mask,
            )

        pooled_pre_projection = pooled

        if self.post_pool_norm:
            pooled_for_projection = self.norm3(
                pooled_pre_projection
            )
        else:
            pooled_for_projection = (
                pooled_pre_projection
            )

        pooled_output = self.time_out_proj(
            self.dropout(
                pooled_for_projection
            )
        )

        if self.mean_residual or self.mean_anchor_residual:
            mean_output = masked_mean_pooling(
                features,
                attention_mask,
                eps=self.eps,
            )

            mean_branch = F.normalize(
                mean_output,
                p=2,
                dim=-1,
                eps=self.eps,
            )
            flag_branch = F.normalize(
                pooled_output,
                p=2,
                dim=-1,
                eps=self.eps,
            )

            self._last_mean_branch = (
                mean_branch.detach()
            )
            self._last_flag_branch = (
                flag_branch.detach()
            )

            if self.mean_residual:
                # Proposal 1.0: convex interpolation.
                #
                # alpha=0 -> exact Mean cosine endpoint.
                # alpha=1 -> exact original-FLaG cosine endpoint.
                alpha = torch.clamp(
                    self.mean_mix_alpha,
                    min=0.0,
                    max=1.0,
                )

                pooled_output = (
                    (1.0 - alpha) * mean_branch
                    + alpha * flag_branch
                )

                self._last_mean_mix_alpha = (
                    alpha.detach()
                )

            else:
                # Proposal 1.1: Mean-anchored orthogonal residual.
                #
                # A naive form normalize(mean + beta * flag) is merely a
                # reparameterization of Proposal 1.0 under cosine scoring.
                # To create a genuinely different hypothesis class, remove
                # the component of FLaG parallel to Mean and let FLaG add
                # only complementary information:
                #
                #   r_perp = f - <f,m> m
                #   z      = normalize(m + beta * r_perp)
                #
                # The Mean direction is therefore never cancelled by the
                # residual branch. beta=0 is exact Mean.
                alignment = (
                    flag_branch
                    * mean_branch
                ).sum(
                    dim=-1,
                    keepdim=True,
                )

                alignment_for_geometry = (
                    alignment.detach()
                    if self.mean_alignment_stopgrad
                    else alignment
                )

                residual = (
                    flag_branch
                    - alignment_for_geometry * mean_branch
                )

                if (
                    self.mean_anchor_unbounded
                    or self.mean_alignment_anchor
                ):
                    beta = self.mean_anchor_beta_unbounded
                else:
                    beta = torch.tanh(
                        self.mean_anchor_beta_raw
                    )

                if self.mean_alignment_anchor:
                    # Alignment-aware anchor:
                    #
                    #   a = <f,m>
                    #   r = f - a m
                    #   z = normalize((1+a)m + beta r)
                    #
                    # If f is nearly orthogonal to m (a≈0), the Mean axis is
                    # retained. If f is nearly opposite to m (a≈-1), the Mean
                    # axis is automatically suppressed. beta=1 reproduces the
                    # direction of m+f exactly; beta=0 is Mean-equivalent
                    # except for the degenerate exact a=-1 case, protected by
                    # a tiny positive floor.
                    mean_axis_scale = (
                        1.0 + alignment_for_geometry
                    ).clamp_min(self.eps)

                    pooled_output = F.normalize(
                        mean_axis_scale * mean_branch
                        + beta * residual,
                        p=2,
                        dim=-1,
                        eps=self.eps,
                    )
                    self._last_mean_axis_scale = (
                        mean_axis_scale.detach()
                    )
                else:
                    pooled_output = F.normalize(
                        mean_branch
                        + beta * residual,
                        p=2,
                        dim=-1,
                        eps=self.eps,
                    )

                self._last_mean_anchor_beta = (
                    beta.detach()
                )
                self._last_orthogonal_residual = (
                    residual.detach()
                )
                self._last_mean_flag_alignment = (
                    alignment.detach()
                )

        if return_pre_projection:
            return (
                pooled_output,
                pooled_pre_projection,
            )

        return pooled_output
    


class STFTLatentAttentionGatePooling(
    FFTLatentAttentionGatePooling
):
    """
    MVP STFT-FLaG.

    Difference from original FLaG:
        global rFFT
        ->
        overlapping local rFFT

    MVP settings:
        - rectangular window
        - n_fft == win_length
        - no explicit frame positional embedding
        - no multi-scale fusion
        - original latent attention / gate / time pooling retained

    Local spectra are flattened as:
        [num_frames, num_freqs, 2*d_model]
        ->
        [num_frames * num_freqs, 2*d_model]

    After frequency modulation:
        reshape
        -> local iFFT
        -> overlap-add
        -> masked time pooling
    """

    def __init__(
        self,
        d_model: int,
        win_length: int = 16,
        hop_length: int = 8,
        use_frame_positional_encoding: bool = False,
        max_frame_positions: int = 128,
        local_window_type: str = "rect",
        center_frames: bool = False,
        **kwargs,
    ):
        if win_length <= 0:
            raise ValueError(
                f"win_length must be > 0, got {win_length}"
            )

        if hop_length <= 0:
            raise ValueError(
                f"hop_length must be > 0, got {hop_length}"
            )

        if hop_length > win_length:
            raise ValueError(
                "MVP STFT expects hop_length <= win_length"
            )

        # No global Hann window here.
        # E2 isolates localization itself.
        super().__init__(
            d_model=d_model,
            window_type=None,
            **kwargs,
        )

        self.win_length = int(win_length)
        self.hop_length = int(hop_length)

        if local_window_type not in {"rect", "hann"}:
            raise ValueError(
                "local_window_type must be 'rect' or 'hann', "
                f"got {local_window_type}"
            )

        self.local_window_type = local_window_type
        self.center_frames = bool(center_frames)

        if (
            self.local_window_type == "hann"
            and not self.center_frames
        ):
            raise ValueError(
                "Hann STFT requires center_frames=True in this MVP "
                "so sentence-boundary tokens are not multiplied by zero."
            )

        self.use_frame_positional_encoding = bool(
            use_frame_positional_encoding
        )

        self.max_frame_positions = int(
            max_frame_positions
        )

        if self.max_frame_positions <= 0:
            raise ValueError(
                "max_frame_positions must be > 0"
            )

        if self.use_frame_positional_encoding:
            # Zero initialization keeps E4 identical to E3
            # at the very beginning of training. The model
            # then learns whether frame identity is useful.
            self.frame_pos_embedding = nn.Parameter(
                torch.zeros(
                    self.max_frame_positions,
                    self.freq_dim,
                )
            )

    def _build_local_frequency_tokens(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        """
        Build local STFT spectral tokens.

        rect + center=False:
            reproduces the previous E3 behavior.

        hann + center=True:
            zero-pad win_length//2 on both sides,
            apply Hann inside every local frame.
        """

        B, T, D = features.shape

        if attention_mask is None:
            mask = torch.ones(
                B,
                T,
                dtype=torch.bool,
                device=features.device,
            )
        else:
            mask = attention_mask.bool()

        lengths = mask.long().sum(dim=1)

        if torch.any(lengths <= 0):
            raise ValueError(
                "Every sample must contain at least one valid token."
            )

        # MVP assumes normal right padding:
        # 111111000...
        positions = torch.arange(
            T,
            device=features.device,
        ).unsqueeze(0)

        expected_mask = (
            positions < lengths.unsqueeze(1)
        )

        if not torch.equal(mask, expected_mask):
            raise ValueError(
                "STFT-FLaG MVP expects contiguous "
                "right-padded attention masks."
            )

        # Padding tokens are zero before STFT.
        x = (
            features
            * mask.unsqueeze(-1).to(features.dtype)
        )

        if self.center_frames:
            left_pad = self.win_length // 2
            right_pad = self.win_length // 2
        else:
            left_pad = 0
            right_pad = 0

        if left_pad > 0 or right_pad > 0:
            left = x.new_zeros(
                B,
                left_pad,
                D,
            )

            right = x.new_zeros(
                B,
                right_pad,
                D,
            )

            x = torch.cat(
                [left, x, right],
                dim=1,
            )

        # -----------------------------------------------------
        # Number of valid frames for each sample
        # -----------------------------------------------------

        if self.center_frames:
            # With win/2 padding on both sides this matches
            # ordinary centered STFT framing for even windows.
            #
            # Example:
            # L=12, win=8, hop=4 -> 4 frames.
            n_frames = (
                torch.div(
                    lengths,
                    self.hop_length,
                    rounding_mode="floor",
                )
                + 1
            )

        else:
            # Previous E3 framing.
            extra = torch.clamp(
                lengths - self.win_length,
                min=0,
            )

            n_frames = (
                torch.div(
                    extra + self.hop_length - 1,
                    self.hop_length,
                    rounding_mode="floor",
                )
                + 1
            )

        max_frames = int(
            n_frames.max().item()
        )

        frame_covered_len = (
            (max_frames - 1)
            * self.hop_length
            + self.win_length
        )

        current_len = x.size(1)

        work_len = max(
            current_len,
            frame_covered_len,
        )

        if work_len > current_len:
            extra_pad = x.new_zeros(
                B,
                work_len - current_len,
                D,
            )

            x = torch.cat(
                [x, extra_pad],
                dim=1,
            )

        # -----------------------------------------------------
        # Extract local frames
        # -----------------------------------------------------

        frames = []

        for frame_idx in range(max_frames):
            start = (
                frame_idx
                * self.hop_length
            )

            end = (
                start
                + self.win_length
            )

            frames.append(
                x[:, start:end, :]
            )

        frames = torch.stack(
            frames,
            dim=1,
        )

        # [B, W]
        frame_ids = torch.arange(
            max_frames,
            device=features.device,
        ).unsqueeze(0)

        frame_valid = (
            frame_ids
            < n_frames.unsqueeze(1)
        )

        # Fake batching-only frames are zero.
        frames = (
            frames
            * frame_valid[
                :, :, None, None
            ].to(frames.dtype)
        )

        # -----------------------------------------------------
        # Local analysis window
        # -----------------------------------------------------

        local_window = self._get_local_window(
            dtype=features.dtype,
            device=features.device,
        )

        # [win] -> [1, 1, win, 1]
        frames = (
            frames
            * local_window[
                None, None, :, None
            ]
        )

        # -----------------------------------------------------
        # Local rFFT
        # -----------------------------------------------------

        spec = torch.fft.rfft(
            frames,
            n=self.win_length,
            dim=2,
        )

        freq = torch.cat(
            [spec.real, spec.imag],
            dim=-1,
        )

        num_freqs = freq.size(2)

        freq_tokens = freq.reshape(
            B,
            max_frames * num_freqs,
            2 * D,
        )

        freq_attention_mask = (
            frame_valid
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                num_freqs,
            )
            .reshape(
                B,
                max_frames * num_freqs,
            )
        )

        meta = {
            "T": T,
            "D": D,
            "max_frames": max_frames,
            "num_freqs": num_freqs,
            "work_len": work_len,
            "frame_valid": frame_valid,
            "left_pad": left_pad,
            "right_pad": right_pad,
            "local_window": local_window,
        }

        self._last_stft_frame_valid = (
            frame_valid.detach()
        )

        self._last_stft_freq_mask = (
            freq_attention_mask.detach()
        )

        self._last_local_window = (
            local_window.detach()
        )

        return (
            freq_tokens,
            freq_attention_mask,
            meta,
        )
    def _add_frame_position_for_attention(
        self,
        freq_tokens: torch.Tensor,
        freq_attention_mask: Optional[torch.Tensor],
        meta,
    ) -> torch.Tensor:
        """
        Add explicit frame identity ONLY to the tokens used by
        latent attention.

        The raw spectral coefficients remain untouched and are
        still used for gating + inverse FFT.
        """

        if not self.use_frame_positional_encoding:
            return freq_tokens

        max_frames = meta["max_frames"]
        num_freqs = meta["num_freqs"]

        if max_frames > self.max_frame_positions:
            raise ValueError(
                f"Need {max_frames} frame positions, but "
                f"max_frame_positions={self.max_frame_positions}"
            )

        # [W, 2D]
        frame_pos = self.frame_pos_embedding[
            :max_frames
        ]

        # Every frequency bin from the same local window
        # receives the same frame-position vector.
        #
        # [W, 2D]
        # ->
        # [W, F, 2D]
        position_tokens = (
            frame_pos[:, None, :]
            .expand(
                max_frames,
                num_freqs,
                self.freq_dim,
            )
        )

        # Flatten in exactly the same [window, frequency] order
        # as freq_tokens.
        #
        # [W, F, 2D]
        # ->
        # [W*F, 2D]
        position_tokens = position_tokens.reshape(
            max_frames * num_freqs,
            self.freq_dim,
        )

        attention_tokens = (
            freq_tokens
            + position_tokens.unsqueeze(0)
        )

        # Fake padded frames should remain invisible.
        if freq_attention_mask is not None:
            attention_tokens = (
                attention_tokens
                * freq_attention_mask
                .unsqueeze(-1)
                .to(attention_tokens.dtype)
            )

        self._last_frame_position_tokens = (
            position_tokens.detach()
        )

        self._last_attention_freq_tokens = (
            attention_tokens.detach()
        )

        return attention_tokens

    def _get_local_window(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:

        if self.local_window_type == "rect":
            return torch.ones(
                self.win_length,
                dtype=dtype,
                device=device,
            )

        if self.local_window_type == "hann":
            # periodic=True is the standard FFT/STFT-style Hann.
            # With win=8 / hop=4 it also has convenient
            # overlap properties.
            return torch.hann_window(
                self.win_length,
                periodic=True,
                dtype=dtype,
                device=device,
            )

        raise RuntimeError(
            f"Unsupported local window: "
            f"{self.local_window_type}"
        )


    def _back_from_local_frequency(
        self,
        enhanced_freq: torch.Tensor,
        meta,
    ) -> torch.Tensor:

        B = enhanced_freq.size(0)

        T = meta["T"]
        D = meta["D"]

        max_frames = meta["max_frames"]
        num_freqs = meta["num_freqs"]
        work_len = meta["work_len"]

        frame_valid = meta["frame_valid"]

        left_pad = meta["left_pad"]

        local_window = meta[
            "local_window"
        ]

        enhanced_freq = (
            enhanced_freq.reshape(
                B,
                max_frames,
                num_freqs,
                2 * D,
            )
        )

        real, imag = enhanced_freq.chunk(
            2,
            dim=-1,
        )

        spec = torch.complex(
            real,
            imag,
        )

        # iFFT returns the analysis-windowed
        # local signal.
        local_time = torch.fft.irfft(
            spec,
            n=self.win_length,
            dim=2,
        )

        # -----------------------------------------------------
        # Synthesis window
        # -----------------------------------------------------

        local_time = (
            local_time
            * local_window[
                None, None, :, None
            ]
        )

        recon = local_time.new_zeros(
            B,
            work_len,
            D,
        )

        window_envelope = (
            local_time.new_zeros(
                B,
                work_len,
                1,
            )
        )

        window_sq = (
            local_window ** 2
        )

        for frame_idx in range(max_frames):

            start = (
                frame_idx
                * self.hop_length
            )

            end = (
                start
                + self.win_length
            )

            valid = (
                frame_valid[
                    :, frame_idx
                ]
                .to(local_time.dtype)
                .view(B, 1, 1)
            )

            recon[:, start:end, :] = (
                recon[:, start:end, :]
                + local_time[
                    :, frame_idx, :, :
                ]
                * valid
            )

            window_envelope[
                :, start:end, :
            ] = (
                window_envelope[
                    :, start:end, :
                ]
                + window_sq[
                    None, :, None
                ]
                * valid
            )

        self._last_window_envelope = (
            window_envelope.detach()
        )

        recon = (
            recon
            / window_envelope.clamp(
                min=1e-8
            )
        )

        # Remove centered zero padding.
        if self.center_frames:
            recon = recon[
                :,
                left_pad:left_pad + T,
                :,
            ]
        else:
            recon = recon[
                :, :T, :
            ]

        return recon

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_pre_projection: bool = False,
    ):
        _, T, D = features.shape

        if D != self.d_model:
            raise ValueError(
                f"Expected last dim {self.d_model}, got {D}"
            )

        (
            freq_tokens,
            freq_attention_mask,
            meta,
        ) = self._build_local_frequency_tokens(
            features,
            attention_mask,
        )

        self._last_seq_len = T

        if self.use_latent:

            attention_freq_tokens = (
                self._add_frame_position_for_attention(
                    freq_tokens,
                    freq_attention_mask,
                    meta,
                )
            )

            latent_out = (
                self._latent_pool_in_frequency(
                    attention_freq_tokens,
                    freq_attention_mask=(
                        freq_attention_mask
                    ),
                )
            )

            # IMPORTANT:
            # gate the RAW spectral coefficients,
            # not position-augmented tokens.
            enhanced_freq = self._apply_gate(
                freq_tokens,
                latent_out,
            )        

        else:
            enhanced_freq = self._apply_gate(
                freq_tokens,
                latent_out=None,
            )

        time_tokens = (
            self._back_from_local_frequency(
                enhanced_freq,
                meta,
            )
        )

        # Explicitly zero padding again after local iFFT.
        if attention_mask is not None:
            time_tokens = (
                time_tokens
                * attention_mask.unsqueeze(-1).to(
                    time_tokens.dtype
                )
            )

        if self.time_pool == "mean":
            pooled = masked_mean_pooling(
                time_tokens,
                attention_mask,
                eps=self.eps,
            )
        else:
            pooled = masked_max_pooling(
                time_tokens,
                attention_mask,
            )

        pooled_pre_projection = pooled

        if self.post_pool_norm:
            pooled_for_projection = self.norm3(
                pooled_pre_projection
            )
        else:
            pooled_for_projection = (
                pooled_pre_projection
            )

        pooled_output = self.time_out_proj(
            self.dropout(
                pooled_for_projection
            )
        )

        if return_pre_projection:
            return (
                pooled_output,
                pooled_pre_projection,
            )

        return pooled_output
