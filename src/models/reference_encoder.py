"""
Reference Encoder with Global Style Tokens (GST).

Based on:
    Wang et al. "Style Tokens: Unsupervised Style Modeling, Control and Transfer
    in End-to-End Speech Synthesis" (ICML 2018).

The reference encoder converts a mel-spectrogram from a reference utterance into
a fixed-size style embedding.  Global Style Tokens (GST) are then obtained by
attending over a bank of trainable token embeddings, producing a compact
representation that can be injected into the synthesis pipeline without any
prosody labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceEncoder(nn.Module):
    """Convolutional stack that maps a mel-spectrogram to a fixed embedding.

    The architecture follows the original GST paper: a stack of strided 2-D
    convolutions reduces the temporal and frequency dimensions, after which the
    resulting feature map is flattened and passed through a GRU to produce a
    single reference embedding vector.

    Args:
        n_mels: Number of mel-filter bins (frequency dimension).
        conv_channels: Output channels for each convolutional layer.
        kernel_size: Kernel size used in all convolutional layers.
        gru_hidden: Hidden size of the final GRU.
        ref_embedding_dim: Dimensionality of the reference embedding output.
    """

    CONV_STRIDES = (2, 2)

    def __init__(
        self,
        n_mels: int = 80,
        conv_channels: tuple[int, ...] = (32, 32, 64, 64, 128, 128),
        kernel_size: int = 3,
        gru_hidden: int = 128,
        ref_embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        in_channels = 1
        layers: list[nn.Module] = []
        for out_channels in conv_channels:
            layers += [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=self.CONV_STRIDES,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
        self.convs = nn.Sequential(*layers)

        # Compute the frequency dimension after all strided convolutions
        freq_dim = n_mels
        for _ in conv_channels:
            freq_dim = (freq_dim + self.CONV_STRIDES[1] - 1) // self.CONV_STRIDES[1]
        gru_input_size = conv_channels[-1] * max(freq_dim, 1)

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden,
            batch_first=True,
        )
        self.linear = nn.Linear(gru_hidden, ref_embedding_dim)

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        """Encode a mel-spectrogram to a reference embedding.

        Args:
            mels: Mel-spectrogram of shape ``(B, n_mels, T)`` or
                ``(B, 1, n_mels, T)``.

        Returns:
            Reference embedding of shape ``(B, ref_embedding_dim)``.
        """
        if mels.dim() == 3:
            mels = mels.unsqueeze(1)  # (B, 1, n_mels, T)

        out = self.convs(mels)  # (B, C, freq', T')
        B, C, F, T = out.shape
        out = out.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)  # (B, T', C*freq')

        _, h_n = self.gru(out)  # h_n: (1, B, gru_hidden)
        ref_emb = self.linear(h_n.squeeze(0))  # (B, ref_embedding_dim)
        return ref_emb


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention used for GST."""

    def __init__(self, query_dim: int, key_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert key_dim % num_heads == 0, "key_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = key_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, key_dim, bias=False)
        self.k_proj = nn.Linear(key_dim, key_dim, bias=False)
        self.v_proj = nn.Linear(key_dim, key_dim, bias=False)
        self.out_proj = nn.Linear(key_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Attend over *key* (style tokens) conditioned on *query* (reference embedding).

        Args:
            query: Tensor of shape ``(B, 1, query_dim)`` — the reference embedding
                treated as a single query token.
            key: Tensor of shape ``(num_tokens, key_dim)`` — the learnable style
                token embeddings (no batch dimension; shared across the batch).

        Returns:
            Style embedding of shape ``(B, query_dim)``.
        """
        B = query.size(0)
        # Expand style tokens to batch
        key = key.unsqueeze(0).expand(B, -1, -1)  # (B, num_tokens, key_dim)
        value = key

        Q = self.q_proj(query)  # (B, 1, key_dim)
        K = self.k_proj(key)    # (B, num_tokens, key_dim)
        V = self.v_proj(value)  # (B, num_tokens, key_dim)

        # Split into heads
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            s = t.size()
            return t.view(s[0], s[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V).permute(0, 2, 1, 3).contiguous()
        out = out.view(B, 1, -1)
        return self.out_proj(out).squeeze(1)  # (B, query_dim)


class GlobalStyleToken(nn.Module):
    """Global Style Token module.

    Combines the :class:`ReferenceEncoder` with a bank of learnable style token
    embeddings and multi-head attention to produce a single style embedding for
    each utterance in a fully unsupervised manner.

    Args:
        n_mels: Mel-spectrogram frequency bins.
        num_tokens: Number of global style tokens.
        token_dim: Dimensionality of each style token.
        num_heads: Number of attention heads.
        ref_embedding_dim: Intermediate reference embedding dimension.
        style_embedding_dim: Final style embedding dimension (= ``token_dim``
            after the attention projection).
        dropout: Dropout probability in attention.
    """

    def __init__(
        self,
        n_mels: int = 80,
        num_tokens: int = 10,
        token_dim: int = 256,
        num_heads: int = 8,
        ref_embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ref_encoder = ReferenceEncoder(
            n_mels=n_mels, ref_embedding_dim=ref_embedding_dim
        )
        # Learnable style token embeddings — shape (num_tokens, token_dim)
        self.style_tokens = nn.Parameter(
            torch.randn(num_tokens, token_dim) * 0.02
        )
        self.attention = MultiHeadAttention(
            query_dim=ref_embedding_dim,
            key_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.style_dim = ref_embedding_dim

    def forward(self, ref_mels: torch.Tensor) -> torch.Tensor:
        """Compute GST-based style embedding for a batch of reference utterances.

        Args:
            ref_mels: Reference mel-spectrograms, shape ``(B, n_mels, T)`` or
                ``(B, 1, n_mels, T)``.

        Returns:
            Style embedding of shape ``(B, ref_embedding_dim)``.
        """
        ref_emb = self.ref_encoder(ref_mels)          # (B, ref_embedding_dim)
        query = ref_emb.unsqueeze(1)                   # (B, 1, ref_embedding_dim)
        style_emb = self.attention(query, self.style_tokens)  # (B, ref_embedding_dim)
        return style_emb
