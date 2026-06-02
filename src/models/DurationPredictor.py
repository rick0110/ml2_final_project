import torch
import torch.nn as nn


class DurationPredictor(nn.Module):
    """Predict per-token duration (number of mel frames) from encoder outputs.

    Simple conv-based predictor that returns positive floats; caller should
    convert to integers (e.g. via rounding or floor) before length-regulation.
    
    Problem: Predicting the number of mel spectrogram frames for each token in a sequence is crucial for text-to-speech models, as it helps determine how long each phoneme should be pronounced.

    Usage: This module takes encoder outputs and predicts durations for each token. It's used during inference to generate mel spectrograms without teacher forcing.
    
    Args:
        input_dim (int): Dimension of the input features from the encoder.
        conv_channels (int, optional): Number of channels in convolutional layers. Default is 256.
        kernel_size (int, optional): Kernel size for convolutional layers. Default is 3.

    Returns:
        torch.Tensor: Predicted durations as positive floats.
    """

    def __init__(self, input_dim: int, conv_channels: int = 256, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.LayerNorm(conv_channels)
        self.norm2 = nn.LayerNorm(conv_channels)
        self.fc = nn.Linear(conv_channels, 1)

    def forward(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        # encoder_outputs: (batch, seq_len, dim)
        x = encoder_outputs.transpose(1, 2)  # (batch, dim, seq_len)
        x = self.conv1(x)
        x = x.transpose(1, 2)  # (batch, seq_len, conv_channels)
        x = self.norm1(x)
        x = self.relu(x)

        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.relu(x)

        # project to scalar per time-step
        out = self.fc(x)  # (batch, seq_len, 1)
        out = out.squeeze(-1)  # (batch, seq_len)
        # Predict positive durations using softplus (returns floats)
        durations = torch.nn.functional.softplus(out)
        return durations
