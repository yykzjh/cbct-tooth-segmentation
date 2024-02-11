from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm

from lib.models.modules.GlobalPMFSBlock import GlobalPMFSBlock_AP_Separate


class ConvBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            strides: int = 1,
            dropout=0.0,
    ):
        super().__init__()
        layers = [
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c: torch.Tensor = self.conv(x)
        return x_c


class UpConv(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size=3, strides=2, dropout=0.0):
        super().__init__()
        self.up = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act="relu",
            adn_ordering="NDA",
            norm=Norm.BATCH,
            dropout=dropout,
            is_transposed=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_u: torch.Tensor = self.up(x)
        return x_u


class AttentionBlock(nn.Module):
    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, dropout=0.0):
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](f_int),
        )

        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[Norm.BATCH, spatial_dims](1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionUNet3D(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Sequence[int] | int = 3,
            up_kernel_size: Sequence[int] | int = 3,
            dropout: float = 0.0,
            with_pmfs_block=False
    ):
        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = (64, 128, 256, 512, 1024)
        self.strides = (2, 2, 2, 2)
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.dropout = dropout

        self.in_conv = ConvBlock(spatial_dims=self.dimensions, in_channels=self.in_channels, out_channels=64, dropout=self.dropout)

        self.enc1 = ConvBlock(spatial_dims=self.dimensions, in_channels=64, out_channels=128, strides=2, dropout=self.dropout)
        self.enc2 = ConvBlock(spatial_dims=self.dimensions, in_channels=128, out_channels=256, strides=2, dropout=self.dropout)
        self.enc3 = ConvBlock(spatial_dims=self.dimensions, in_channels=256, out_channels=512, strides=2, dropout=self.dropout)
        self.enc4 = ConvBlock(spatial_dims=self.dimensions, in_channels=512, out_channels=1024, strides=2, dropout=self.dropout)

        self.with_pmfs_block = with_pmfs_block
        if with_pmfs_block:
            self.global_pmfs_block = GlobalPMFSBlock_AP_Separate(
                in_channels=[64, 128, 256, 512, 1024],
                max_pool_kernels=[16, 8, 4, 2, 1],
                ch=48,
                ch_k=48,
                ch_v=48,
                br=5,
                dim="3d"
            )

        self.attn1 = AttentionBlock(spatial_dims=self.dimensions, f_g=64, f_l=64, f_int=32)
        self.attn2 = AttentionBlock(spatial_dims=self.dimensions, f_g=128, f_l=128, f_int=64)
        self.attn3 = AttentionBlock(spatial_dims=self.dimensions, f_g=256, f_l=256, f_int=128)
        self.attn4 = AttentionBlock(spatial_dims=self.dimensions, f_g=512, f_l=512, f_int=256)

        self.upconv1 = UpConv(spatial_dims=self.dimensions, in_channels=128, out_channels=64, strides=2, kernel_size=self.up_kernel_size)
        self.upconv2 = UpConv(spatial_dims=self.dimensions, in_channels=256, out_channels=128, strides=2, kernel_size=self.up_kernel_size)
        self.upconv3 = UpConv(spatial_dims=self.dimensions, in_channels=512, out_channels=256, strides=2, kernel_size=self.up_kernel_size)
        self.upconv4 = UpConv(spatial_dims=self.dimensions, in_channels=1024, out_channels=512, strides=2, kernel_size=self.up_kernel_size)

        self.dec1 = Convolution(spatial_dims=self.dimensions, in_channels=128, out_channels=64, dropout=self.dropout)
        self.dec2 = Convolution(spatial_dims=self.dimensions, in_channels=256, out_channels=128, dropout=self.dropout)
        self.dec3 = Convolution(spatial_dims=self.dimensions, in_channels=512, out_channels=256, dropout=self.dropout)
        self.dec4 = Convolution(spatial_dims=self.dimensions, in_channels=1024, out_channels=512, dropout=self.dropout)

        self.out_conv = Convolution(
            spatial_dims=self.dimensions,
            in_channels=64,
            out_channels=self.out_channels,
            kernel_size=1,
            strides=1,
            padding=0,
            conv_only=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        if self.with_pmfs_block:
            x5 = self.global_pmfs_block([x1, x2, x3, x4, x5])

        g4 = self.upconv4(x5)
        m4 = self.attn4(g=g4, x=x4)
        d4 = self.dec4(torch.cat((m4, g4), dim=1))
        g3 = self.upconv3(d4)
        m3 = self.attn3(g=g3, x=x3)
        d3 = self.dec3(torch.cat((m3, g3), dim=1))
        g2 = self.upconv2(d3)
        m2 = self.attn2(g=g2, x=x2)
        d2 = self.dec2(torch.cat((m2, g2), dim=1))
        g1 = self.upconv1(d2)
        m1 = self.attn1(g=g1, x=x1)
        d1 = self.dec1(torch.cat((m1, g1), dim=1))

        out = self.out_conv(d1)
        return out


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    x = torch.randn((1, 1, 64, 64, 64)).to(device)

    model = AttentionUNet3D(spatial_dims=3, in_channels=1, out_channels=2, with_pmfs_block=True).to(device)

    output = model(x)

    print(x.size())
    print(output.size())
