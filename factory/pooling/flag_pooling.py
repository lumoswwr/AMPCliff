"""FLaG core pooling: FFT latent attention gate (release-only module)."""

from typing import Optional

import torch
import torch.nn as nn

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

        self.time_out_proj = nn.Linear(d_model, d_model)
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

        gate = torch.sigmoid(
            self.freq_gate(gate_input)
        )

        self._last_raw_gate = gate.detach()

        if self.gate_residual:
            gate = 1.0 + gate

        enhanced_freq = (
            freq_tokens * gate.unsqueeze(1)
        )

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
