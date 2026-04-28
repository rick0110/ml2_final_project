"""
Variance Adaptor.

Predicts and injects frame-level acoustic variance (pitch, energy, duration)
into the content representation.  Based on FastSpeech 2:

    Ren et al. "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"
    (ICLR 2021).

The duration predictor is used to up-sample the content features from the
encoder frame-rate to the decoder frame-rate via a length regulator.
Pitch and energy embeddings are added on top of the up-sampled sequence so
that the decoder receives a fully conditioned representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariancePredictor(nn.Module):
    """Convolutional variance predictor (shared architecture for pitch, energy,
    and duration predictors).

    Args:
        d_model: Input feature dimension.
        num_conv_layers: Number of 1-D convolutional layers.
        kernel_size: Kernel size for the convolutional layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_conv_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_conv_layers):
            layers += [
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.ReLU(inplace=True),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            ]
        self.convs = nn.Sequential(*layers)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a scalar variance per frame.

        Args:
            x: Encoder hidden states, shape ``(B, T, d_model)``.

        Returns:
            Predicted scalars, shape ``(B, T, 1)``.
        """
        # Conv1d expects (B, C, T)
        out = x.transpose(1, 2)  # (B, d_model, T)
        # Apply each layer; LayerNorm needs (B, T, C) so transpose around it
        for layer in self.convs:
            if isinstance(layer, nn.LayerNorm):
                out = layer(out.transpose(1, 2)).transpose(1, 2)
            else:
                out = layer(out)
        out = out.transpose(1, 2)  # (B, T, d_model)
        return self.linear(out)  # (B, T, 1)


class LengthRegulator(nn.Module):
    """Expand encoder frames to decoder frames using predicted durations.

    The regulator repeats each encoder frame according to the integer duration
    assigned to that frame (nearest-integer rounding of the predicted value).

    Args:
        max_len: Maximum output sequence length (clips if exceeded).
    """

    def __init__(self, max_len: int = 2048) -> None:
        super().__init__()
        self.max_len = max_len

    def forward(
        self,
        x: torch.Tensor,
        durations: torch.Tensor,
        target_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Regulate the sequence length.

        Args:
            x: Input features, shape ``(B, T_enc, d_model)``.
            durations: Integer durations per encoder frame, shape ``(B, T_enc)``.
                Values are expected to be non-negative integers.
            target_len: If provided, the output is padded / trimmed to exactly
                this length (used during teacher-forced training).

        Returns:
            A tuple ``(regulated_x, output_lengths)`` where *regulated_x* has
            shape ``(B, T_dec, d_model)`` and *output_lengths* is ``(B,)``.
        """
        durations = durations.long().clamp(min=0)
        outputs = []
        lengths = durations.sum(dim=1)  # (B,)

        max_out_len = int(lengths.max().item())
        if target_len is not None:
            max_out_len = target_len
        max_out_len = min(max_out_len, self.max_len)

        for b in range(x.size(0)):
            expanded = torch.repeat_interleave(x[b], durations[b], dim=0)
            cur_len = expanded.size(0)
            if cur_len < max_out_len:
                pad = x.new_zeros(max_out_len - cur_len, x.size(2))
                expanded = torch.cat([expanded, pad], dim=0)
            else:
                expanded = expanded[:max_out_len]
            outputs.append(expanded)

        return torch.stack(outputs, dim=0), lengths


class VarianceAdaptor(nn.Module):
    """Combines duration, pitch, and energy predictors with a length regulator.

    The style embedding from the Reference Encoder is added to the content
    features before variance prediction so that prosody variance is conditioned
    on the reference style.

    Args:
        d_model: Dimension of input content features.
        style_dim: Dimension of the style embedding (from GST).
        num_conv_layers: Number of conv layers in each variance predictor.
        kernel_size: Conv kernel size in predictors.
        dropout: Dropout probability.
        max_len: Maximum output length for the length regulator.
    """

    def __init__(
        self,
        d_model: int = 256,
        style_dim: int = 128,
        num_conv_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        # Project style embedding to model dimension for addition
        self.style_proj = nn.Linear(style_dim, d_model)

        self.duration_predictor = VariancePredictor(d_model, num_conv_layers, kernel_size, dropout)
        self.pitch_predictor = VariancePredictor(d_model, num_conv_layers, kernel_size, dropout)
        self.energy_predictor = VariancePredictor(d_model, num_conv_layers, kernel_size, dropout)

        self.pitch_embedding = nn.Linear(1, d_model)
        self.energy_embedding = nn.Linear(1, d_model)

        self.length_regulator = LengthRegulator(max_len=max_len)

    def forward(
        self,
        x: torch.Tensor,
        style_emb: torch.Tensor,
        target_durations: torch.Tensor | None = None,
        target_pitch: torch.Tensor | None = None,
        target_energy: torch.Tensor | None = None,
        target_len: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Apply variance adaption to content features.

        Args:
            x: Content features from mapping network, shape ``(B, T_enc, d_model)``.
            style_emb: Style embedding from GST, shape ``(B, style_dim)``.
            target_durations: Ground-truth integer durations, shape ``(B, T_enc)``.
                When provided the length regulator uses these instead of the
                predicted durations (teacher-forcing).
            target_pitch: Ground-truth pitch values, shape ``(B, T_enc, 1)``.
            target_energy: Ground-truth energy values, shape ``(B, T_enc, 1)``.
            target_len: Target output length for length regulator.

        Returns:
            Dictionary with keys:
                - ``"output"``: Variance-adapted features ``(B, T_dec, d_model)``.
                - ``"pred_durations"``: Predicted log durations ``(B, T_enc, 1)``.
                - ``"pred_pitch"``: Predicted pitch ``(B, T_enc, 1)``.
                - ``"pred_energy"``: Predicted energy ``(B, T_enc, 1)``.
                - ``"output_lengths"``: Actual output lengths ``(B,)``.
        """
        # Inject style into content representation
        style_projected = self.style_proj(style_emb).unsqueeze(1)  # (B, 1, d_model)
        x = x + style_projected

        # Predict variance quantities
        pred_durations = self.duration_predictor(x)  # (B, T_enc, 1)
        pred_pitch = self.pitch_predictor(x)          # (B, T_enc, 1)
        pred_energy = self.energy_predictor(x)        # (B, T_enc, 1)

        # Select target or predicted pitch / energy
        pitch_input = target_pitch if target_pitch is not None else pred_pitch
        energy_input = target_energy if target_energy is not None else pred_energy
        x = x + self.pitch_embedding(pitch_input) + self.energy_embedding(energy_input)

        # Compute durations for length regulation
        if target_durations is not None:
            durations = target_durations
        else:
            # Use rounded exponential of predicted log-durations (log scale predictor)
            durations = torch.clamp(pred_durations.squeeze(-1).exp().round(), min=0)

        regulated, out_lengths = self.length_regulator(x, durations, target_len=target_len)

        return {
            "output": regulated,
            "pred_durations": pred_durations,
            "pred_pitch": pred_pitch,
            "pred_energy": pred_energy,
            "output_lengths": out_lengths,
        }
