import torch
from torch import nn
from typing import Union, Tuple


class GST(nn.Module):
    """Global Style Tokens module for TTS. Consists of a stack of convolutional layers followed by a GRU.
    
    Problem: This module captures global style information from mel-spectrograms and provides it as an embedding vector to the model, allowing for stylistic control over generated speech.

    Usage: The GST is used in conjunction with other TTS models (like CrossAttentionTTS) to provide stylistic variation during synthesis. It can be fine-tuned or pre-trained on specific styles or speaker voices.
    
    Args:
        n_conv_layers (int, optional): Number of convolutional layers. Default is 6.
        hidden_size (int, optional): Size of the GRU hidden state. Default is 128.
        n_style_tokens (int, optional): Number of style tokens to generate. Default is 10.
        n_mels (int, optional): Dimensionality of mel-spectrogram features. Default is 80.
        n_heads (int, optional): Number of attention heads for multi-head attention mechanism. Default is 4.

    Returns:
        torch.Tensor (or Tuple[torch.Tensor, torch.Tensor]): Style embedding vector (and optionally attention weights).
    """
    
    def __init__(self, n_conv_layers: int = 6, hidden_size: int = 128, n_style_tokens: int = 10, n_mels: int = 80, n_heads: int = 4):
        super().__init__()
        assert n_conv_layers%2 == 0, "n_conv_layers must be even"
        self.n_conv_layers = n_conv_layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(out_channels=32*(2**i), in_channels=32*(2**(i-1)) if i > 0 else 1, kernel_size=(3, 3), stride=(2, 2), padding=1),
                nn.SiLU(),
                nn.BatchNorm2d(32*(2**i)),
                nn.Conv2d(out_channels=32*(2**i), in_channels=32*(2**i), kernel_size=(3, 3), stride=(2, 2), padding=1),
                nn.SiLU(),
                nn.BatchNorm2d(32*(2**i)),
            ]) for i in range(n_conv_layers//2)
        ]) # -> [batch_size, 32*(2**(n_conv_layers//2 - 1)), n_mels/(2**(n_conv_layers//2)), time_steps/(2**(n_conv_layers//2))]

        with torch.no_grad():
            style_shape_probe = torch.zeros(1, 1, n_mels, 1)
            style_shape_probe = self.forward_n_conv_layers(style_shape_probe)
            style_attention_input_size = style_shape_probe.size(1) * style_shape_probe.size(2)

        self.style_attention = nn.GRU(input_size=style_attention_input_size, hidden_size=hidden_size, batch_first=True)
        
        self.multHeadAttention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=n_heads, batch_first=True)
        self.style_tokens = nn.Parameter(torch.randn(n_style_tokens, hidden_size)) # -> [n_style_tokens, hidden_size]

    def forward(self, x: torch.Tensor, return_att_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.forward_style_embedding(x, return_att_weights)
    
    def forward_style_embedding(self, x: torch.Tensor, return_att_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.forward_n_conv_layers(x)
        x = x.reshape(x.size(0), -1, x.size(3)) # -> [batch_size, 32*(2**(n_conv_layers//2 - 1)) * n_mels/(2**(n_conv_layers//2)), time_steps'] -> preserve time resolution
        x = x.transpose(1, 2) # -> [batch_size, time_steps', feature_dim]
        x = self.style_attention(x) # -> [1, batch_size, hidden_size]
        _, x = x # -> [1, batch_size, hidden_size]
        x = x.squeeze(0) # -> [batch_size, hidden_size]
        x = torch.nn.functional.silu(x) # -> [batch_size, hidden_size]
        return self.forward_style_multihead_attention(x, return_att_weights)

    def forward_n_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.conv_layers:
            x = conv(x)
        return x
    
    def forward_style_multihead_attention(self, x: torch.Tensor, return_att_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        query = x.unsqueeze(1) # -> [batch_size, 1, hidden_size]
        keys = self.style_tokens.unsqueeze(0).expand(x.size(0), -1, -1)
        values = self.style_tokens.unsqueeze(0).expand(x.size(0), -1, -1)
        att_out, att_weights = self.multHeadAttention(query, keys, values) # -> [batch_size, 1, hidden_size]
        
        if return_att_weights:
            return att_out, att_weights.squeeze(1) # -> [batch_size, 1, hidden_size], [batch_size, n_style_tokens]
        else:
            return att_out # -> [batch_size, 1, hidden_size]