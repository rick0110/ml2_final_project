"""
Mapping Network.

Bridges the HuBERT content encoder output space and the VITS decoder input
space.  The mapping network is the primary trainable component when HuBERT is
frozen; it learns to project the self-supervised speech representations into
the decoder's expected feature space using only a small Portuguese corpus.

Architecture:
    Linear → LayerNorm → ReLU (× num_layers) → Linear → LayerNorm
    Each hidden layer also receives the style embedding via a FiLM-like
    affine conditioning (scale and shift) so that the mapping is
    style-aware from the very first stage of the pipeline.
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioning layer.

    Applies a learned affine transform to *x* conditioned on *condition*:
        out = scale * x + shift
    where *scale* and *shift* are predicted from *condition*.

    Args:
        feature_dim: Dimension of the feature vector to be modulated.
        condition_dim: Dimension of the conditioning vector.
    """

    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.scale_proj = nn.Linear(condition_dim, feature_dim)
        self.shift_proj = nn.Linear(condition_dim, feature_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            x: Features, shape ``(B, T, feature_dim)``.
            condition: Conditioning vector, shape ``(B, condition_dim)``.

        Returns:
            Modulated features, shape ``(B, T, feature_dim)``.
        """
        scale = self.scale_proj(condition).unsqueeze(1)  # (B, 1, feature_dim)
        shift = self.shift_proj(condition).unsqueeze(1)  # (B, 1, feature_dim)
        return scale * x + shift


class MappingBlock(nn.Module):
    """A single residual block in the mapping network.

    Consists of a linear projection, layer norm, FiLM conditioning, and
    a residual connection.

    Args:
        d_model: Hidden dimension.
        style_dim: Style embedding dimension for FiLM conditioning.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, style_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.film = FiLMLayer(d_model, style_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear(x)
        out = self.norm(out)
        out = self.film(out, style_emb)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class MappingNetwork(nn.Module):
    """Intermediate mapping network: HuBERT space → VITS decoder space.

    Transforms frame-level HuBERT representations into the feature space
    expected by the VITS-style decoder.  The network is composed of an
    input projection followed by a stack of FiLM-conditioned residual blocks
    and an output projection.

    Args:
        input_dim: Dimension of HuBERT features (default 768 for base model).
        output_dim: Dimension of features expected by the decoder.
        hidden_dim: Hidden dimension of the intermediate blocks.
        num_layers: Number of residual mapping blocks.
        style_dim: Style embedding dimension (from GST) for FiLM conditioning.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        style_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [MappingBlock(hidden_dim, style_dim, dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, content: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        """Map HuBERT content features to the decoder feature space.

        Args:
            content: HuBERT features, shape ``(B, T, input_dim)``.
            style_emb: Style embedding from GST, shape ``(B, style_dim)``.

        Returns:
            Mapped features, shape ``(B, T, output_dim)``.
        """
        out = self.input_proj(content)  # (B, T, hidden_dim)
        for block in self.blocks:
            out = block(out, style_emb)
        return self.output_proj(out)  # (B, T, output_dim)
