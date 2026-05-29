"""Direct text-to-mel TTS model used by the direct TTS pipeline.

This module is intentionally independent from the train_first_step and
train_try_2 model code. The model is text-only: it predicts a mel
spectrogram from tokenized text and does not consume audio features as an
input. A length predictor provides inference-time frame estimates so the
decoder can generate a spectrogram without teacher forcing.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim=embedding_dim, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be 2D (batch, seq_len), got {tuple(token_ids.shape)}")

        padding_mask = token_ids.eq(self.pad_idx)
        x = self.embedding(token_ids)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        x = self.final_norm(x)

        valid_mask = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        pooled = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
        return x, padding_mask, pooled


class LengthPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_text_state: torch.Tensor) -> torch.Tensor:
        raw = self.net(pooled_text_state).squeeze(-1)
        return F.softplus(raw) + 1.0


class MelDecoder(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, n_layers: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim=model_dim, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        return self.final_norm(x)


class DirectTTSModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 1024,
        n_mels: int = 80,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.pad_idx = pad_idx

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=model_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            pad_idx=pad_idx,
        )
        self.length_predictor = LengthPredictor(model_dim, hidden_dim=model_dim, dropout=dropout)
        self.decoder_projection = nn.Linear(model_dim, model_dim)
        self.decoder = MelDecoder(model_dim=model_dim, n_heads=n_heads, n_layers=n_layers, ff_dim=ff_dim, dropout=dropout)
        self.output_projection = nn.Linear(model_dim, n_mels)
        self.refinement = nn.Sequential(
            nn.Conv1d(n_mels, n_mels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(n_mels, n_mels, kernel_size=3, padding=1),
        )

    def _expand_sequence(self, sequence: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if sequence.dim() != 3:
            raise ValueError(f"sequence must be 3D (batch, seq_len, hidden), got {tuple(sequence.shape)}")

        expanded_sequences = []
        padded_masks = []
        target_lengths = lengths.to(dtype=torch.long).clamp_min(1)
        for item, length in zip(sequence, target_lengths.tolist()):
            if item.size(0) == length:
                expanded = item
            else:
                expanded = F.interpolate(
                    item.transpose(0, 1).unsqueeze(0),
                    size=int(length),
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).transpose(0, 1)
            expanded_sequences.append(expanded)
            padded_masks.append(torch.zeros(int(length), dtype=torch.bool, device=sequence.device))

        padded = nn.utils.rnn.pad_sequence(expanded_sequences, batch_first=True)
        mask = nn.utils.rnn.pad_sequence(padded_masks, batch_first=True, padding_value=True)
        return padded, mask

    def _compute_decoder_lengths(
        self,
        predicted_lengths: torch.Tensor,
        target_lengths: torch.Tensor | None,
        target_mel: torch.Tensor | None,
    ) -> torch.Tensor:
        if target_lengths is not None:
            return target_lengths.to(device=predicted_lengths.device, dtype=torch.long).clamp_min(1)
        if target_mel is not None:
            return torch.full(
                (predicted_lengths.size(0),),
                int(target_mel.size(-1)),
                device=predicted_lengths.device,
                dtype=torch.long,
            )
        return predicted_lengths.round().to(dtype=torch.long).clamp_min(1)

    def _forward_impl(
        self,
        text_ids: torch.Tensor,
        target_mel: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if text_ids.dim() != 2:
            raise ValueError(f"text_ids must be 2D (batch, seq_len), got {tuple(text_ids.shape)}")
        if target_mel is not None and target_mel.dim() != 3:
            raise ValueError(f"target_mel must be 3D (batch, n_mels, time), got {tuple(target_mel.shape)}")

        text_states, text_padding_mask, pooled_text_state = self.text_encoder(text_ids)
        predicted_lengths = self.length_predictor(pooled_text_state)
        decoder_lengths = self._compute_decoder_lengths(predicted_lengths, target_lengths, target_mel)

        expanded_states, expanded_padding_mask = self._expand_sequence(
            self.decoder_projection(text_states),
            decoder_lengths,
        )
        decoder_states = self.decoder(expanded_states, padding_mask=expanded_padding_mask)

        predicted_mel = self.output_projection(decoder_states).transpose(1, 2)
        predicted_mel = self.refinement(predicted_mel) + predicted_mel

        return {
            "text_states": text_states,
            "text_padding_mask": text_padding_mask,
            "pooled_text_state": pooled_text_state,
            "predicted_lengths": predicted_lengths,
            "decoder_lengths": decoder_lengths.to(dtype=torch.float32),
            "expanded_states": expanded_states,
            "expanded_padding_mask": expanded_padding_mask,
            "decoder_states": decoder_states,
            "predicted_mel": predicted_mel,
        }

    def forward(
        self,
        text_ids: torch.Tensor,
        target_mel: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        states = self._forward_impl(text_ids=text_ids, target_mel=target_mel, target_lengths=target_lengths)
        return states["predicted_mel"], states["predicted_lengths"]

    def forward_with_intermediates(
        self,
        text_ids: torch.Tensor,
        target_mel: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        states = self._forward_impl(text_ids=text_ids, target_mel=target_mel, target_lengths=target_lengths)
        return states["predicted_mel"], states["predicted_lengths"], states

    def get_trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def load_direct_tts_model(
    vocab_size: int,
    model_dim: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    ff_dim: int = 1024,
    n_mels: int = 80,
    pad_idx: int = 0,
    dropout: float = 0.1,
) -> DirectTTSModel:
    return DirectTTSModel(
        vocab_size=vocab_size,
        model_dim=model_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        n_mels=n_mels,
        pad_idx=pad_idx,
        dropout=dropout,
    )


def get_model_size_info(model: DirectTTSModel) -> dict:
    return {
        "text_encoder": sum(parameter.numel() for parameter in model.text_encoder.parameters()),
        "length_predictor": sum(parameter.numel() for parameter in model.length_predictor.parameters()),
        "decoder": sum(parameter.numel() for parameter in model.decoder.parameters()),
        "output_projection": sum(parameter.numel() for parameter in model.output_projection.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.get_trainable_parameters()),
        "total": sum(parameter.numel() for parameter in model.parameters()),
    }