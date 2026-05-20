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
        self.attention_layer = nn.ModuleList([nn.Conv1d(embedding_dim, embedding_dim, kernel_size=K_attention_window, padding=K_attention_window//2, bias=False) for i in range(n_time_steps)])


    def forward(self, x):
        """x shape is (batch_size, seq_len, embedding_dim)"""
        x = self.embedding(x) # (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        for attention in self.attention_layer:
            x = attention(x) # (batch_size, embedding_dim, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, embedding_dim)
        return x
    