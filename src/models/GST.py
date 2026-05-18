import torch
from torch import nn


class GST(nn.Module):
    '''Global Style Tokens module for TTS. Consists of a stack of convolutional layers followed by a GRU.
    expected input shape: (batch_size, 1-> channels, n_mels, time_steps)
    output shape: (batch_size, 128) style embedding

    '''
    def __init__(self, n_conv_layers: int = 6, hidden_size: int = 128, n_style_tokens: int = 10, n_mels: int = 80, n_heads: int = 4):
        super().__init__()
        assert n_conv_layers%2 == 0, "n_conv_layers must be even"
        self.n_conv_layers = n_conv_layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(out_channels=32*(2**i), in_channels=32*(2**(i-1)) if i > 0 else 1, kernel_size=(3, 3), stride=(2, 2), padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(32*(2**i)),
                nn.Conv2d(out_channels=32*(2**i), in_channels=32*(2**i), kernel_size=(3, 3), stride=(2, 2), padding=1),
                nn.ReLU(inplace=True),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_style_embedding(x)
        
        return x
    
    def forward_style_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_n_conv_layers(x)
        x = x.reshape(x.size(0), -1, x.size(3)) # -> [batch_size, 32*(2**(n_conv_layers//2 - 1)) * n_mels/(2**(n_conv_layers//2)), time_steps'] -> preserve time resolution
        x = x.transpose(1, 2) # -> [batch_size, time_steps', feature_dim]
        x = self.style_attention(x) # -> [1, batch_size, hidden_size]
        _, x = x # -> [1, batch_size, hidden_size]
        x = x.squeeze(0) # -> [batch_size, hidden_size]
        x = torch.tanh(x) # -> [batch_size, hidden_size]
        x = self.forward_style_multihead_attention(x) # -> [batch_size, hidden_size]
        return x

    def forward_n_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.conv_layers:
            x = conv(x)
        return x
    
    def forward_style_multihead_attention(self, x: torch.Tensor) -> torch.Tensor:
        query = x.unsqueeze(1) # -> [batch_size, 1, hidden_size]
        keys = self.style_tokens.unsqueeze(0).expand(x.size(0), -1, -1)
        values = self.style_tokens.unsqueeze(0).expand(x.size(0), -1, -1)
        att_out, att_weights = self.multHeadAttention(query, keys, values) # -> [batch_size, hidden_size]
        return att_out, att_weights
    


        


