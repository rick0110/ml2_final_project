"""
Global Style Token (GST) module.

Responsibilities:
    - Capture global style information from mel-spectrograms.
    - Provide style embeddings for TTS models to control prosody/identity.
    - Implement a bank of learnable style tokens with attention mechanism.

Main Classes:
    - GST: The primary Global Style Token module.

Tensor Conventions:
    B = batch size
    T = sequence length (frames)
    n_mels = mel frequency bins
    H = hidden dimension
    N_tokens = number of style tokens
"""
import torch
from torch import nn
from torch import Tensor
from typing import Union, Tuple, List, Optional

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

    Architecture:
        Conv Stack (Feature Extraction) -> GRU (Temporal Encoding) -> Multi-Head Attention (Style Selection)

    Inputs:
        x:
            Shape (B, n_mels, T)
        return_att_weights:
            Whether to return attention weights.

    Outputs:
        style_embedding:
            Shape (B, 1, H)
        attention_weights (optional):
            Shape (B, N_tokens)

    Example:
        >>> gst = GST(n_conv_layers=6, hidden_size=128, n_style_tokens=10, n_mels=80, n_heads=4)
        >>> mel_input = torch.randn(16, 80, 100) # (B, n_mels, T)
        >>> style_embedding, attention_weights = gst(mel_input, return_att_weights=True)
    """

    def __init__(
        self,
        n_conv_layers: int = DEFAULT_N_CONV_LAYERS,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        n_style_tokens: int = DEFAULT_N_STYLE_TOKENS,
        n_mels: int = DEFAULT_N_MELS,
        n_heads: int = DEFAULT_N_HEADS,
    ) -> None:
        """
        Initialize the GST module.

        Args:
            n_conv_layers (int): Number of convolutional layers. Must be even.
            hidden_size (int): Size of the GRU hidden state and style tokens.
            n_style_tokens (int): Number of learnable style tokens in the bank.
            n_mels (int): Dimensionality of input mel-spectrogram features.
            n_heads (int): Number of attention heads for the multi-head attention mechanism.
        """
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

        # Determine input size for GRU by probing the output shape of conv layers
        with torch.no_grad():
            # Create a dummy input to infer shape after convolutions
            dummy_input: Tensor = torch.zeros(1, 1, n_mels, 1)
            # Pass through convolutional layers
            conv_output_shape_probe: Tensor = self.forward_n_conv_layers(dummy_input)
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
        self, x: Tensor, return_att_weights: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass of the GST module.

        Args:
            x (Tensor): Input mel-spectrogram. Shape: (B, n_mels, T)
            return_att_weights (bool): If True, also returns attention weights.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Style embedding vector (B, 1, H) 
                and optionally attention weights (B, N_tokens).
        """
        return self.forward_style_embedding(x, return_att_weights)

    def forward_style_embedding(
        self, x: Tensor, return_att_weights: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Computes the style embedding and optionally attention weights.

        Args:
            x (Tensor): Input mel-spectrogram. Shape: (B, n_mels, T)
            return_att_weights (bool): If True, returns attention weights.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Style embedding and optionally attention weights.
        """
        # Ensure input has channel dimension if needed
        if x.ndim == 3:
            x = x.unsqueeze(1) # (B, 1, n_mels, T)

        # Pass through convolutional layers
        x = self.forward_n_conv_layers(x) # (B, C_last, n_mels', T')
        
        # Reshape and transpose for GRU input
        # Flatten channels * n_mels' for GRU input feature dimension
        x = x.reshape(x.size(0), -1, x.size(3))  # Shape: (B, C_last * n_mels', T')
        x = x.transpose(1, 2)  # Shape: (B, T', C_last * n_mels')

        # Process with GRU
        gru_output: Tensor
        gru_output, _ = self.style_attention(x) # Shape: (B, T', H)
        
        # Use the last hidden state
        x = gru_output[:, -1, :] # Shape: (B, H)
        x = torch.nn.functional.silu(x) # Shape: (B, H)

        # Compute style embedding using multi-head attention over style tokens
        return self.forward_style_multihead_attention(x, return_att_weights)

    def forward_n_conv_layers(self, x: Tensor) -> Tensor:
        """
        Applies the sequence of convolutional layers.

        Args:
            x (Tensor): Input tensor. Shape: (B, 1, n_mels, T)

        Returns:
            Tensor: Output tensor after convolutional layers. Shape: (B, C_last, n_mels', T')
        """
        for conv in self.conv_layers:
            x = conv(x) # (B, C_i, n_mels_i, T_i)
        return x

    def forward_style_multihead_attention(
        self, x: Tensor, return_att_weights: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Computes the final style embedding using multi-head attention over style tokens.

        Args:
            x (Tensor): Processed features from GRU. Shape: (B, H)
            return_att_weights (bool): If True, returns attention weights.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Style embedding and optionally attention weights.
        """
        # Query is the GRU output, expanded to have a time dimension of 1
        query: Tensor = x.unsqueeze(1)  # Shape: (B, 1, H)
        
        # Keys and Values are the learnable style tokens, broadcasted to match batch size
        keys: Tensor = self.style_tokens.unsqueeze(0).expand(x.size(0), -1, -1)   # (B, N_tokens, H)
        values: Tensor = self.style_tokens.unsqueeze(0).expand(x.size(0), -1, -1) # (B, N_tokens, H)

        # Apply multi-head attention
        att_out: Tensor
        att_weights: Tensor
        att_out, att_weights = self.multHeadAttention(query, keys, values) # (B, 1, H), (B, 1, N_tokens)

        if return_att_weights:
            return att_out, att_weights.squeeze(1)  # (B, 1, H), (B, N_tokens)
        else:
            return att_out  # (B, 1, H)
