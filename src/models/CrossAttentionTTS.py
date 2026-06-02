"""Cross-attention TTS model used by the second training pipeline.

Pipeline:
Text -> text transformer -> text states
Mel -> mel transformer -> mel states
Mel -> GST -> style token
Text states <-> mel states -> cross attention fusion
Fused states + style token -> temporal cross attention
Temporal states -> mel projection

Problem: This model aims to capture both the textual and acoustic information for generating high-quality speech synthesis. It uses a combination of text and audio encoders, along with global style tokens (GST) to provide stylistic control over the generated output.

Usage: The CrossAttentionTTSModel class is used in training pipelines where both text and mel-spectrogram inputs are available during training. This model can be fine-tuned for different styles or speaker voices by adjusting the GST embeddings.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GST import GST
from models.TextEncoder import TextEncoderMultiHeadAttention


class PositionalEncoding(nn.Module):
    """Add positional encoding to input embeddings for transformer models."""
    
    def __init__(self, embedding_dim: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network."""
    
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
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
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class SequenceEncoder(nn.Module):
    """Sequence encoder for CrossAttentionTTS model."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=hidden_dim,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for fusing text and mel states."""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query, context, context, need_weights=False)
        x = self.norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class CrossAttentionTTSModel(nn.Module):
    """Cross-attention TTS model that predicts mel spectrograms from text and audio."""
    
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 1024,
        style_embedding_dim: int = 128,
        n_mels: int = 80,
    ):
        super().__init__()
        self.n_mels = n_mels

        self.text_encoder = TextEncoderMultiHeadAttention(
            vocab_size=vocab_size,
            embedding_dim=model_dim,
            n_heads=n_heads,
            n_steps=n_layers,
            ff_dim=ff_dim,
            dropout=0.0,
        )
        self.mel_encoder = SequenceEncoder(
            input_dim=n_mels,
            hidden_dim=model_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=0.1,
        )
        self.style_extractor = GST(
            n_conv_layers=4,
            hidden_size=style_embedding_dim,
            n_style_tokens=30,
            n_mels=n_mels,
            n_heads=n_heads,
        )

        self.text_to_mel_cross_attention = CrossAttentionBlock(model_dim, n_heads, ff_dim)
        self.mel_to_text_cross_attention = CrossAttentionBlock(model_dim, n_heads, ff_dim)
        self.temporal_cross_attention = CrossAttentionBlock(model_dim, n_heads, ff_dim)

        self.style_projection = nn.Sequential(
            nn.Linear(style_embedding_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, model_dim),
        )
        self.output_projection = nn.Linear(model_dim, n_mels)

    def _forward_impl(
        self,
        text_ids: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if target_mel.dim() != 3:
            raise ValueError(f"target_mel must be 3D (batch, n_mels, time), got {tuple(target_mel.shape)}")

        text_states = self.text_encoder(text_ids)
        mel_states = self.mel_encoder(target_mel.transpose(1, 2))
        style_tokens = self.style_extractor(target_mel.unsqueeze(1))

        text_context = self.text_to_mel_cross_attention(text_states, mel_states)
        mel_context = self.mel_to_text_cross_attention(mel_states, text_states)
        if text_context.size(1) != mel_context.size(1):
            text_context = F.interpolate(
                text_context.transpose(1, 2),
                size=mel_context.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        fused_states = self.fusion(torch.cat([text_context, mel_context], dim=-1))

        style_context = self.style_projection(style_tokens).unsqueeze(1).expand(-1, fused_states.size(1), -1)
        temporal_states = self.temporal_cross_attention(fused_states, style_context)
        predicted_mel = self.output_projection(temporal_states)

        return {
            "text_states": text_states,
            "mel_states": mel_states,
            "style_tokens": style_tokens,
            "text_context": text_context,
            "mel_context": mel_context,
            "fused_states": fused_states,
            "style_context": style_context,
            "temporal_states": temporal_states,
            "predicted_mel": predicted_mel,
        }

    def forward(
        self,
        text_ids: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        states = self._forward_impl(text_ids=text_ids, target_mel=target_mel)
        return states["predicted_mel"], states["style_tokens"]

    def forward_with_intermediates(
        self,
        text_ids: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        states = self._forward_impl(text_ids=text_ids, target_mel=target_mel)
        return states["predicted_mel"], states["style_tokens"], states

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def load_cross_attention_tts_model(
    vocab_size: int,
    model_dim: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    ff_dim: int = 1024,
    style_embedding_dim: int = 128,
    n_mels: int = 80,
) -> CrossAttentionTTSModel:
    return CrossAttentionTTSModel(
        vocab_size=vocab_size,
        model_dim=model_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        style_embedding_dim=style_embedding_dim,
        n_mels=n_mels,
    )


def get_model_size_info(model: CrossAttentionTTSModel) -> dict:
    return {
        "text_encoder": sum(p.numel() for p in model.text_encoder.parameters()),
        "mel_encoder": sum(p.numel() for p in model.mel_encoder.parameters()),
        "style_extractor": sum(p.numel() for p in model.style_extractor.parameters()),
        "trainable": sum(p.numel() for p in model.get_trainable_parameters()),
        "total": sum(p.numel() for p in model.parameters()),
    }
