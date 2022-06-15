from torch import nn, Tensor
from typing import List, Tuple


class ResidualBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        # Residual Path
        self.channel_matching = nn.Identity()
        if in_channels != out_channels:
            self.channel_matching =  nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

        # Convolutional layers
        padding = (kernel_size-1)//2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.channel_matching(x)
        out += residual
        out = self.relu(out)
        return out


class ResidualConv(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        pooling: List[int] = [2,2],
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.resconv = ResidualBlock(in_channels, out_channels, kernel_size)
        self.avgpooling = nn.AvgPool2d(pooling)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.resconv(x)
        out = self.dropout(out)
        out = self.avgpooling(out)
        return out


class ResidualConvTranspose(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        scale: Tuple[int] = (2,2),
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.resconv = ResidualBlock(in_channels, out_channels, kernel_size)
        self.upsampling = nn.Upsample(scale_factor=scale, mode='nearest')
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsampling(x)
        out = self.resconv(out)
        out = self.dropout(out)
        return out
