import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
from typing import Tuple, Union


class AddCoords(nn.Module):
    """
    Module that appends spatial coordinates to a tensor.
    
    This is used to provide spatial awareness to convolutions (CoordConv).
    Currently, only rank 2 (2D spatial dimensions) is supported.
    
    Args:
        rank (int): The spatial rank of the tensor (e.g., 2 for 2D images).
        with_r (bool, optional): Whether to also append a radial coordinate representing
            distance from the center. Defaults to False.
            
    Example:
        >>> add_coords = AddCoords(rank=2, with_r=True)
        >>> input_tensor = torch.randn(2, 3, 32, 32)
        >>> output = add_coords(input_tensor)
        >>> output.shape
        torch.Size([2, 6, 32, 32])
    """
    def __init__(self, rank: int, with_r: bool = False):
        super().__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass appending coordinate channels.
        
        Args:
            input_tensor (torch.Tensor): Input tensor. Shape: (batch_size, channels, dim_y, dim_x)
            
        Returns:
            torch.Tensor: Tensor with concatenated coordinate channels. Shape: (batch_size, channels + 2 + int(with_r), dim_y, dim_x)
        """
        if self.rank == 2:
            batch_size_shape, _, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.float32, device=input_tensor.device)  # [1, 1, 1, dim_x]
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.float32, device=input_tensor.device)  # [1, 1, 1, dim_y]

            xx_range = torch.arange(dim_y, dtype=torch.float32, device=input_tensor.device)  # [dim_y]
            yy_range = torch.arange(dim_x, dtype=torch.float32, device=input_tensor.device)  # [dim_x]
            xx_range = xx_range[None, None, :, None]  # [1, 1, dim_y, 1]
            yy_range = yy_range[None, None, :, None]  # [1, 1, dim_x, 1]

            xx_channel = torch.matmul(xx_range, xx_ones)  # [1, 1, dim_y, dim_x]
            yy_channel = torch.matmul(yy_range, yy_ones)  # [1, 1, dim_x, dim_y]
            yy_channel = yy_channel.permute(0, 1, 3, 2)  # [1, 1, dim_y, dim_x]

            xx_channel = xx_channel.float() / (dim_y - 1)  # [1, 1, dim_y, dim_x]
            yy_channel = yy_channel.float() / (dim_x - 1)  # [1, 1, dim_y, dim_x]
            xx_channel = xx_channel * 2 - 1  # [1, 1, dim_y, dim_x]
            yy_channel = yy_channel * 2 - 1  # [1, 1, dim_y, dim_x]

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)  # [batch_size, 1, dim_y, dim_x]
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)  # [batch_size, 1, dim_y, dim_x]

            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)  # [batch_size, channels + 2, dim_y, dim_x]
            if self.with_r:
                rr = torch.sqrt(
                    torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2)
                )  # [batch_size, 1, dim_y, dim_x]
                out = torch.cat([out, rr], dim=1)  # [batch_size, channels + 3, dim_y, dim_x]
            return out

        raise NotImplementedError(f"Rank {self.rank} not supported")


class CoordConv2d(conv.Conv2d):
    """
    2D Convolution layer with coordinate channels added to the input.
    
    This allows the convolutional filters to learn spatial dependence.
    
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
        with_r (bool, optional): If True, adds an extra channel with the radial distance. Default: False
        
    Example:
        >>> coord_conv = CoordConv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        >>> input_tensor = torch.randn(2, 3, 32, 32)
        >>> output = coord_conv(input_tensor)
        >>> output.shape
        torch.Size([2, 16, 32, 32])
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
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv2d(
            in_channels + self.rank + int(with_r),
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the CoordConv layer.
        
        Args:
            input_tensor (torch.Tensor): The input tensor. Shape: (batch_size, in_channels, dim_y, dim_x)
            
        Returns:
            torch.Tensor: The convolved output tensor. Shape: (batch_size, out_channels, out_dim_y, out_dim_x)
        """
        out = self.addcoords(input_tensor)  # [batch_size, in_channels + rank + int(with_r), dim_y, dim_x]
        return self.conv(out)  # [batch_size, out_channels, out_dim_y, out_dim_x]
