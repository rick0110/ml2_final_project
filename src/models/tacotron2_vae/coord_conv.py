"""
Coordinate Convolutional layers.

Responsibilities:
    - Implement AddCoords to augment input tensors with spatial coordinates.
    - Implement CoordConv2d as a replacement for standard Conv2d to provide spatial awareness.

Main Classes:
    - AddCoords: Appends coordinate channels (x, y, and optionally r) to a 4D tensor.
    - CoordConv2d: Convolutional layer that utilizes AddCoords.

Tensor Conventions:
    B = batch size
    C = number of channels
    H = height (y dimension)
    W = width (x dimension)
"""
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.modules.conv as conv
from typing import Tuple, Union


class AddCoords(nn.Module):
    """
    Module that appends spatial coordinates to a tensor.
    
    This is used to provide spatial awareness to convolutions (CoordConv).
    Currently, only rank 2 (2D spatial dimensions) is supported.
    
    Architecture:
        Concatenates normalized coordinate grids (-1 to 1) to the channel dimension.
    
    Inputs:
        input_tensor:
            Shape (B, C, H, W)
            
    Outputs:
        output_tensor:
            Shape (B, C + 2 + int(with_r), H, W)
            
    Example:
        >>> add_coords = AddCoords(rank=2, with_r=True)
        >>> x = torch.randn(2, 3, 32, 32)
        >>> y = add_coords(x)
    """
    def __init__(self, rank: int, with_r: bool = False) -> None:
        """
        Initialize the AddCoords module.

        Args:
            rank (int): The spatial rank of the tensor (only 2 supported).
            with_r (bool): Whether to append a radial coordinate channel.
        """
        super().__init__()
        self.rank: int = rank
        self.with_r: bool = with_r

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Forward pass appending coordinate channels.
        
        Args:
            input_tensor (Tensor): Input feature map.
                Shape: (B, C, H, W)
            
        Returns:
            Tensor: Feature map with additional coordinate channels.
                Shape: (B, C + 2 + int(with_r), H, W)
        """
        if self.rank == 2:
            batch_size_shape, _, dim_y, dim_x = input_tensor.shape
            
            # Create coordinate grids
            xx_ones: Tensor = torch.ones([1, 1, 1, dim_x], dtype=torch.float32, device=input_tensor.device)  # (1, 1, 1, W)
            yy_ones: Tensor = torch.ones([1, 1, 1, dim_y], dtype=torch.float32, device=input_tensor.device)  # (1, 1, 1, H)

            xx_range: Tensor = torch.arange(dim_y, dtype=torch.float32, device=input_tensor.device)  # (H,)
            yy_range: Tensor = torch.arange(dim_x, dtype=torch.float32, device=input_tensor.device)  # (W,)
            xx_range = xx_range[None, None, :, None]  # (1, 1, H, 1)
            yy_range = yy_range[None, None, :, None]  # (1, 1, W, 1)

            xx_channel: Tensor = torch.matmul(xx_range, xx_ones)  # (1, 1, H, W)
            yy_channel: Tensor = torch.matmul(yy_range, yy_ones)  # (1, 1, W, H)
            yy_channel = yy_channel.permute(0, 1, 3, 2)  # (1, 1, H, W)

            # Normalize to [-1, 1]
            denom_y = max(dim_y - 1, 1)
            denom_x = max(dim_x - 1, 1)
            xx_channel = xx_channel.float() / denom_y # (1, 1, H, W)
            yy_channel = yy_channel.float() / denom_x # (1, 1, H, W)
            xx_channel = xx_channel * 2 - 1              # (1, 1, H, W)
            yy_channel = yy_channel * 2 - 1              # (1, 1, H, W)

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)  # (B, 1, H, W)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)  # (B, 1, H, W)

            out: Tensor = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)  # (B, C + 2, H, W)
            
            if self.with_r:
                rr: Tensor = torch.sqrt(
                    torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2)
                )  # (B, 1, H, W)
                out = torch.cat([out, rr], dim=1)  # (B, C + 3, H, W)
            
            return out # (B, C + 2 + int(with_r), H, W)

        raise NotImplementedError(f"Rank {self.rank} not supported")


class CoordConv2d(nn.Module):
    """
    2D Convolution layer with coordinate channels added to the input.
    
    This allows the convolutional filters to learn spatial dependence.
    
    Architecture:
        AddCoords -> Conv2d
    
    Inputs:
        input_tensor:
            Shape (B, in_channels, H, W)
            
    Outputs:
        output_tensor:
            Shape (B, out_channels, H_out, W_out)
            
    Example:
        >>> conv = CoordConv2d(in_channels=3, out_channels=16, kernel_size=3)
        >>> x = torch.randn(2, 3, 32, 32)
        >>> y = conv(x)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        with_r: bool = False,
    ) -> None:
        """
        Initialize the CoordConv2d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Union[int, Tuple[int, int]]): Convolution kernel size.
            stride (Union[int, Tuple[int, int]]): Stride.
            padding (Union[int, Tuple[int, int]]): Padding.
            dilation (Union[int, Tuple[int, int]]): Dilation.
            groups (int): Number of groups.
            bias (bool): Whether to use bias.
            with_r (bool): Whether to append radial coordinate channel.
        """
        super().__init__()
        self.rank: int = 2
        self.addcoords: AddCoords = AddCoords(self.rank, with_r)
        self.conv: nn.Conv2d = nn.Conv2d(
            in_channels + self.rank + int(with_r),
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Forward pass applying the CoordConv layer.
        
        Args:
            input_tensor (Tensor): Input feature map.
                Shape: (B, in_channels, H, W)
            
        Returns:
            Tensor: Convolved output.
                Shape: (B, out_channels, H_out, W_out)
        """
        out: Tensor = self.addcoords(input_tensor)  # (B, in_channels + 2 + int(with_r), H, W)
        output: Tensor = self.conv(out)             # (B, out_channels, H_out, W_out)
        return output
