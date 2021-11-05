import typing as tp
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Conv2d -> BatchNorm2d -> ReLU
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: nn.Module = None,
                 **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True) if activation is None else activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ShiftedConvBlock(ConvBlock):
    """
    BatchNorm2d -> ReLU -> Conv2d
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: nn.Module = None,
                 **kwargs):
        super().__init__(in_channels, out_channels, activation, **kwargs)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class ConvTransposeBlock(nn.Module):
    """
    ConvTranspose2d -> BatchNorm2d -> LeakyReLU
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: nn.Module = None,
                 kernel_size: tp.Tuple[int, int] = (3, 3),
                 **kwargs):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(inplace=True) if activation is None else activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
