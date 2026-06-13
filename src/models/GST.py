"""
Global Style Token (GST) module.

This module implements the Global Style Token (GST) mechanism, which aims to
capture global style information from mel-spectrograms and provide it as an
embedding vector to TTS models. This allows for stylistic control over generated
speech, such as speaker identity or emotional tone.

The GST module consists of:
1. A stack of convolutional layers to extract features from mel-spectrograms.
2. A GRU layer to encode temporal information from the extracted features.
3. A Multi-Head Attention mechanism that uses the GRU output as a query to attend
   over learnable style tokens.

Dependencies:
    - torch: PyTorch for neural network operations.
    - torch.nn: PyTorch neural network modules.
    - typing: For type hinting.

Typical Usage:
    >>> import torch
    >>> gst = GST(n_conv_layers=6, hidden_size=128, n_style_tokens=10, n_mels=80, n_heads=4)
    >>> mel_spectrogram = torch.randn(16, 80, 100) # Batch size, n_mels, time_steps
    >>> style_embedding, attention_weights = gst(mel_spectrogram, return_att_weights=True)
    >>> print(style_embedding.shape) # torch.Size([16, 1, 128])
    >>> print(attention_weights.shape) # torch.Size([16, 10])
"""
import torch
from torch import nn
from typing import Union, Tuple, List

# Define default values for hyperparameters
DEFAULT_N_CONV_LAYERS: int = 6
DEFAULT_HIDDEN_SIZE: int = 128
DEFAULT_N_STYLE_TOKENS: int = 10
DEFAULT_N_MELS: int = 80
DEFAULT_N_HEADS: int = 4
DEFAULT_CONV_CHANNELS_INIT: int = 32


class GST(nn.Module):
    """
    Global Style Tokens module for TTS.

    Captures global style information from mel-spectrograms and provides it as an
    embedding vector to the model, enabling stylistic control over generated speech.

    Args:
        n_conv_layers (int, optional): Number of convolutional layers. Must be even.
                                       Defaults to 6.
        hidden_size (int, optional): Size of the GRU hidden state. Defaults to 128.
        n_style_tokens (int, optional): Number of learnable style tokens. Defaults to 10.
        n_mels (int, optional): Dimensionality of mel-spectrogram features. Defaults to 80.
        n_heads (int, optional): Number of attention heads for the multi-head attention mechanism.
                                 Defaults to 4.

    Attributes:
        n_conv_layers (int): Number of convolutional layers.
        conv_layers (nn.ModuleList): List of sequential convolutional blocks.
        style_attention (nn.GRU): GRU layer to process features after convolutions.
        multHeadAttention (nn.MultiheadAttention): Multi-head attention mechanism.
        style_tokens (nn.Parameter): Learnable style tokens.

    Example:
        >>> gst = GST(n_conv_layers=6, hidden_size=128, n_style_tokens=10, n_mels=80, n_heads=4)
        >>> mel_input = torch.randn(16, 80, 100) # Batch size, n_mels, time_steps
        >>> style_embedding, attention_weights = gst(mel_input, return_att_weights=True)
        >>> print(style_embedding.shape)
        torch.Size([16, 1, 128])
        >>> print(attention_weights.shape)
        torch.Size([16, 10])
    """

    def __init__(
        self,
        n_conv_layers: int = DEFAULT_N_CONV_LAYERS,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        n_style_tokens: int = DEFAULT_N_STYLE_TOKENS,
        n_mels: int = DEFAULT_N_MELS,
        n_heads: int = DEFAULT_N_HEADS,
    ):
        super().__init__()
        assert n_conv_layers % 2 == 0, "n_conv_layers must be even"
        self.n_conv_layers: int = n_conv_layers

        # Define convolutional layers
        self.conv_layers: nn.ModuleList = nn.ModuleList()
        for i in range(n_conv_layers // 2):
            # Calculate channel dimensions, doubling for each subsequent block
            out_channels_i: int = DEFAULT_CONV_CHANNELS_INIT * (2**i)
            in_channels_i: int = DEFAULT_CONV_CHANNELS_INIT * (2**(i - 1)) if i > 0 else 1

            conv_block = nn.Sequential(
                # First Conv2d layer
                nn.Conv2d(
                    in_channels=in_channels_i,
                    out_channels=out_channels_i,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                ),
                nn.SiLU(),  # Activation function
                nn.BatchNorm2d(num_features=out_channels_i),  # Batch normalization
                # Second Conv2d layer (same channels)
                nn.Conv2d(
                    in_channels=out_channels_i,
                    out_channels=out_channels_i,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                ),
                nn.SiLU(),
                nn.BatchNorm2d(num_features=out_channels_i),
            )
            self.conv_layers.append(conv_block)
        # After conv_layers, the shape is approximately:
        # [batch_size, 32*(2**(n_conv_layers//2 - 1)), n_mels/(2**(n_conv_layers//2)), time_steps/(2**(n_conv_layers//2))]

        # Determine input size for GRU by probing the output shape of conv layers
        with torch.no_grad():
            # Create a dummy input to infer shape after convolutions
            dummy_input: torch.Tensor = torch.zeros(1, 1, n_mels, 1)
            # Pass through convolutional layers
            conv_output_shape_probe: torch.Tensor = self.forward_n_conv_layers(dummy_input)
            # Calculate the flattened feature dimension
            style_attention_input_size: int = conv_output_shape_probe.size(1) * conv_output_shape_probe.size(2)

        # GRU layer to process the flattened features over time
        self.style_attention: nn.GRU = nn.GRU(
            input_size=style_attention_input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Multi-Head Attention layer
        self.multHeadAttention: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_heads, batch_first=True
        )
        # Learnable style tokens (keys and values for attention)
        self.style_tokens: nn.Parameter = nn.Parameter(
            torch.randn(n_style_tokens, hidden_size)
        )  # Shape: [n_style_tokens, hidden_size]

    def forward(
        self, x: torch.Tensor, return_att_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the GST module.

        Args:
            x (torch.Tensor): Input mel-spectrogram. Shape: (batch_size, n_mels, time_steps)
            return_att_weights (bool, optional): If True, also returns attention weights.
                                                  Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If return_att_weights is False:
                    Style embedding vector. Shape: (batch_size, 1, hidden_size)
                - If return_att_weights is True:
                    Tuple containing:
                        - Style embedding vector. Shape: (batch_size, 1, hidden_size)
                        - Attention weights. Shape: (batch_size, n_style_tokens)
        """
        return self.forward_style_embedding(x, return_att_weights)

    def forward_style_embedding(
        self, x: torch.Tensor, return_att_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the style embedding and optionally attention weights.

        Args:
            x (torch.Tensor): Input mel-spectrogram. Shape: (batch_size, n_mels, time_steps)
            return_att_weights (bool, optional): If True, returns attention weights. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Style embedding and optionally attention weights.
        """
        # Pass through convolutional layers
        x = self.forward_n_conv_layers(x)
        # Reshape and transpose for GRU input
        # x shape after convs: [batch_size, channels, n_mels', time_steps']
        # Flatten channels * n_mels' for GRU input feature dimension
        x = x.reshape(x.size(0), -1, x.size(3))  # Shape: [batch_size, flattened_features, time_steps']
        x = x.transpose(1, 2)  # Shape: [batch_size, time_steps', flattened_features]

        # Process with GRU. GRU output: (num_layers*num_directions, batch, hidden_size)
        gru_output, _ = self.style_attention(x)
        # Use the last hidden state: [batch_size, hidden_size]
        x = gru_output.squeeze(0)
        x = torch.nn.functional.silu(x)  # Apply SiLU activation

        # Compute style embedding using multi-head attention over style tokens
        return self.forward_style_multihead_attention(x, return_att_weights)

    def forward_n_conv_layers(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the sequence of convolutional layers.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, 1, n_mels, time_steps)

        Returns:
            torch.Tensor: Output tensor after convolutional layers.
        """
        for conv in self.conv_layers:
            x = conv(x)
        return x

    def forward_style_multihead_attention(
        self, x: torch.Tensor, return_att_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the final style embedding using multi-head attention over style tokens.

        Args:
            x (torch.Tensor): Processed features from GRU. Shape: (batch_size, hidden_size)
            return_att_weights (bool, optional): If True, returns attention weights. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Style embedding and optionally attention weights.
        """
        # Query is the GRU output, expanded to have a time dimension of 1
        query: torch.Tensor = x.unsqueeze(1)  # Shape: [batch_size, 1, hidden_size]
        # Keys and Values are the learnable style tokens, broadcasted to match batch size
        keys: torch.Tensor = self.style_tokens.unsqueeze(0).expand(x.size(0), -1, -1)
        values: torch.Tensor = self.style_tokens.unsqueeze(0).expand(x.size(0), -1, -1)

        # Apply multi-head attention
        # att_out shape: [batch_size, 1, hidden_size]
        # att_weights shape: [batch_size, 1, n_style_tokens]
        att_out, att_weights = self.multHeadAttention(query, keys, values)

        if return_att_weights:
            # Squeeze the time dimension from attention weights and return
            return att_out, att_weights.squeeze(1)  # Shapes: [batch_size, 1, hidden_size], [batch_size, n_style_tokens]
        else:
            return att_out  # Shape: [batch_size, 1, hidden_size]
