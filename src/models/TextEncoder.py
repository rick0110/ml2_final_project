import math

import torch
import torch.nn as nn

class TextEncoder(torch.nn.Module):
    """Receive the tokenized text and output the corresponding embeddings.
    and the output shape is (batch_size, seq_len, embedding_dim)."""
    def __init__(self, vocab_size: int, embedding_dim: int = 128, K_attention_window: int = 7, n_time_steps: int = 3):
        """k_attention_window should be an odd number to maintain the same sequence length after convolution."""
        assert K_attention_window % 2 == 1, "K_attention_window should be an odd number."
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention_layer = nn.ModuleList([nn.Sequential(nn.Conv1d(embedding_dim, embedding_dim, kernel_size=K_attention_window, padding=K_attention_window//2, bias=False), nn.ReLU(inplace=True)) for i in range(n_time_steps)])


    def forward(self, x):
        """x shape is (batch_size, seq_len, embedding_dim)"""
        x = self.embedding(x) # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        for attention in self.attention_layer:
            x = attention(x) # (batch_size, embedding_dim, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, embedding_dim)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)  # (1, max_len, embedding_dim)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TransformerBlock(nn.Module):
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
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class TextEncoderMultiHeadAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        n_heads: int = 4,
        n_steps: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(n_steps)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return x
    