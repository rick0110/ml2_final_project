import torch
from torch import nn

class GST(torch.nn.Module):
    '''Global Style Tokens module for TTS. Consists of a stack of convolutional layers followed by a GRU.
    expected input shape: (batch_size, 1-> channels, n_mels, time_steps)

    '''
    def __init__(self, n_conv_layers: int = 6):
        super().__init__()
        assert n_conv_layers%2 == 0, "n_conv_layers must be even"
        self.n_conv_layers = n_conv_layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential([
                nn.Conv2d(out_channels=32*(2**i), in_channels=32*(2**(i-1)) if i > 0 else 1, kernel_size=(3, 3), stride=(2, 2), padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32*(2**i)),
                nn.Conv2d(out_channels=32*(2**i), in_channels=32*(2**i), kernel_size=(3, 3), stride=(2, 2), padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32*(2**i)),
            ]) for i in range(n_conv_layers//2)
        ]) # -> [batch_size, 32*(2**(n_conv_layers//2 - 1)), n_mels/(2**(n_conv_layers//2)), time_steps/(2**(n_conv_layers//2))]

        self.gru = nn.GRU(input_size=32*(2**(n_conv_layers//2 - 1)), hidden_size=128, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_n_conv_layers(x)
        x = x.reshape(x.size(0), -1, x.size(3)) # -> [batch_size, 32*(2**(n_conv_layers//2 - 1)), time_steps'] -> preserve time resolution
        x = x.transpose(1, 2) # -> [batch_size, time_steps', feature_dim]
        x = self.gru(x) # -> [1, batch_size, 128]
        x = x[1].squeeze(0) # -> [batch_size, 128]
        x = nn.Tanh()(x) # style embedding typically passed through tanh
        return x
    
    def forward_style_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_n_conv_layers(x)
        x = x.reshape(x.size(0), -1, x.size(3)) # -> [batch_size, 32*(2**(n_conv_layers//2 - 1)), time_steps'] -> preserve time resolution
        x = x.transpose(1, 2) # -> [batch_size, time_steps', feature_dim]
        _, h_n = self.gru(x) # -> [1, batch_size, 128]
        return h_n.squeeze(0) # -> [batch_size, 128]
    
    def forward_n_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.conv_layers:
            x = conv(x)
        return x

