# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/5/15 17:37
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import torch
import torch.nn as nn

from lib.models.modules.GlobalPMFSBlock import GlobalPMFSBlock_AP_Separate



class DenseVoxelNet(nn.Module):
    """
    Implementation based on https://arxiv.org/abs/1708.00573
    Trainable params: 1,783,408 (roughly 1.8 mentioned in the paper)
    """

    def __init__(self, in_channels=1, classes=3, with_pmfs_block=False):
        super(DenseVoxelNet, self).__init__()
        num_input_features = 16
        self.dense_1_out_features = 160
        self.dense_2_out_features = 304
        self.up_out_features = 64
        self.classes = classes
        self.in_channels = in_channels

        self.conv_init = nn.Conv3d(in_channels, num_input_features, kernel_size=1, stride=2, padding=0, bias=False)
        self.dense_1 = _DenseBlock(num_layers=12, num_input_features=num_input_features, bn_size=1, growth_rate=12)
        self.trans = _Transition(self.dense_1_out_features, self.dense_1_out_features)
        self.dense_2 = _DenseBlock(num_layers=12, num_input_features=self.dense_1_out_features, bn_size=1,
                                   growth_rate=12)

        self.with_pmfs_block = with_pmfs_block
        if with_pmfs_block:
            self.global_pmfs_block = GlobalPMFSBlock_AP_Separate(
                in_channels=[160, 160, 304],
                max_pool_kernels=[2, 1, 1],
                ch=48,
                ch_k=48,
                ch_v=48,
                br=3,
                dim="3d"
            )

        self.up_block = _Upsampling(self.dense_2_out_features, self.up_out_features)
        self.conv_final = nn.Conv3d(self.up_out_features, classes, kernel_size=1, padding=0, bias=False)
        self.transpose = nn.ConvTranspose3d(self.dense_1_out_features, self.up_out_features, kernel_size=2, padding=0,
                                            output_padding=0,
                                            stride=2)

    def forward(self, x):
        # Main network path
        x = self.conv_init(x)
        x1 = self.dense_1(x)
        x2, t = self.trans(x1)
        x3 = self.dense_2(x2)

        if self.with_pmfs_block:
            x3 = self.global_pmfs_block([x1, x2, x3])

        x = self.up_block(x3)
        y1 = self.conv_final(x)

        # Auxiliary mid-layer prediction, kind of long-skip connection
        # t = self.transpose(t)
        # y2 = self.conv_final(t)
        return y1


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0.2):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),

        self.drop_rate = drop_rate
        if self.drop_rate > 0:
            self.drop_layer = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = self.drop_layer(new_features)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """
    to keep the spatial dims o=i, this formula is applied
    o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
    """

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate=0.2):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        norm = nn.BatchNorm3d(num_input_features)
        relu = nn.ReLU(inplace=True)
        conv3d = nn.Conv3d(num_input_features, num_output_features,
                           kernel_size=1, padding=0, stride=1)
        self.conv = nn.Sequential(norm, relu, conv3d)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        k = self.conv(x)
        y = self.max_pool(k)
        return y, k


class _Upsampling(nn.Sequential):
    """
    For transpose conv
    o = output, p = padding, k = kernel_size, s = stride, d = dilation
    o = (i -1)*s - 2*p + k + output_padding = (i-1)*2 +2 = 2*i
    """

    def __init__(self, input_features, out_features):
        super(_Upsampling, self).__init__()
        self.tr_conv1_features = 128  # defined in the paper
        self.tr_conv2_features = out_features
        self.add_module('norm', nn.BatchNorm3d(input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(input_features, input_features,
                                          kernel_size=1, stride=1, padding=0, bias=False))

        # Transposed convolutions must be un-padded?
        self.add_module('transp_conv_1',
                        nn.ConvTranspose3d(input_features, self.tr_conv1_features, kernel_size=2, padding=0,
                                           output_padding=0, stride=2))
        self.add_module('transp_conv_2',
                        nn.ConvTranspose3d(self.tr_conv1_features, self.tr_conv2_features, kernel_size=2, padding=0,
                                           output_padding=0, stride=2))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    x = torch.randn((1, 1, 32, 32, 32)).to(device)

    model = DenseVoxelNet(in_channels=1, classes=35, with_pmfs_block=True).to(device)

    output = model(x)

    print(x.size())
    print(output.size())

