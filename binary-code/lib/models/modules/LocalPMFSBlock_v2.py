# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/3 20:02
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))

import torch
import torch.nn as nn

from collections import OrderedDict

from lib.models.modules.PolarizedSelfAttention3d import SequentialPolarizedSelfAttention3d



class LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendInnerProductVector(nn.Module):
    """
    使用多尺度感受野信息扩充内积向量长度的局部极化多尺度感受野自注意力模块
    """
    def __init__(self, in_channel, channel, kernels=[1, 3, 5], group=1, dilation=1, r=4):
        """
        定义一个使用多尺度感受野信息扩充内积向量长度的局部极化多尺度感受野自注意力模块

        :param in_channel: 输入通道数
        :param channel: 输出通道数
        :param kernels: 不同分支的内核大小
        :param group: 分组卷积的组数
        :param dilation: 空洞率
        :param r: 衰减率
        """
        super(LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendInnerProductVector, self).__init__()
        self.inner_c = channel // (len(kernels) * r)
        # 先定义一个卷积层变换输出通道数
        self.basic_conv = nn.Conv3d(in_channels=in_channel, out_channels=channel, kernel_size=3, padding=1)
        # 定义Wq的不同感受野分支的卷积
        self.ch_Wq_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, self.inner_c, kernel_size=k, padding=((k - 1) * dilation) // 2, dilation=dilation, groups=group)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])
        # 定义Wk的不同感受野分支的卷积
        self.ch_Wk_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, 1, kernel_size=k, padding=((k - 1) * dilation) // 2, dilation=dilation)),
                ("bn", nn.BatchNorm3d(1)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])
        # 定义通道注意力的Softmax
        self.ch_softmax = nn.Softmax(1)
        # 定义通道注意力对分数矩阵的卷积Wz
        self.ch_Wz = nn.Conv3d(self.inner_c, channel, kernel_size=1)
        # 定义通道注意力的层归一化
        self.layer_norm = nn.LayerNorm((channel, 1, 1, 1))
        # 定义对注意力分数矩阵的Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

        # 定义Wq的不同感受野分支的卷积
        self.sp_Wq_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, self.inner_c, kernel_size=k, padding=((k - 1) * dilation) // 2, dilation=dilation, groups=group)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])
        # 定义Wk的不同感受野分支的卷积
        self.sp_Wk_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, self.inner_c, kernel_size=k, padding=((k - 1) * dilation) // 2, dilation=dilation, groups=group)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])
        # 定义空间自注意力的Softmax
        self.sp_softmax = nn.Softmax(-1)



    def forward(self, x):
        # 先变换通道数
        x = self.basic_conv(x)
        # 获得输入特征图维度信息
        bs, c, d, h, w = x.size()

        # 计算通道Wq各分支
        ch_Wq_outs = [
            conv(x)
            for conv in self.ch_Wq_convs
        ]
        # 堆叠通道Wq
        ch_Wq = torch.stack(ch_Wq_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 计算通道Wk各分支
        ch_Wk_outs = [
            conv(x)
            for conv in self.ch_Wk_convs
        ]
        # 堆叠通道Wk
        ch_Wk = torch.stack(ch_Wk_outs, dim=-1)  # bs, 1, d, h, w, k
        # 转换维度
        ch_Wq = ch_Wq.reshape(bs, self.inner_c, -1)  # bs, self.inner_c, d*h*w*k
        ch_Wk = ch_Wk.reshape(bs, -1, 1)  # bs, d*h*w*k, 1
        # 进行Softmax处理
        ch_Wk = self.ch_softmax(ch_Wk)  # bs, d*h*w*k, 1
        # 矩阵相乘
        ch_Wz = torch.matmul(ch_Wq, ch_Wk).unsqueeze(-1).unsqueeze(-1)  # bs, self.inner_c, 1, 1, 1
        # 计算通道注意力分数矩阵
        ch_score = self.sigmoid(self.layer_norm(self.ch_Wz(ch_Wz)))  # bs, c, 1, 1, 1
        # 通道增强
        ch_out = ch_score * x

        # 计算空间Wq各分支
        sp_Wq_outs = [
            conv(ch_out)
            for conv in self.sp_Wq_convs
        ]
        # 堆叠空间Wq
        sp_Wq = torch.stack(sp_Wq_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 计算空间Wk各分支
        sp_Wk_outs = [
            conv(ch_out)
            for conv in self.sp_Wk_convs
        ]
        # 堆叠空间Wk
        sp_Wk = torch.stack(sp_Wk_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 转换维度
        sp_Wq = sp_Wq.reshape(bs, -1, d * h * w)  # bs, k*self.inner_c, d*h*w
        sp_Wk = sp_Wk.mean(-1).mean(-1).mean(-1).reshape(bs, 1, -1)  # bs, 1, k*self.inner_c
        # 进行Softmax处理
        sp_Wk = self.sp_softmax(sp_Wk)  # bs, 1, k*self.inner_c
        # 矩阵相乘
        sp_Wz = torch.matmul(sp_Wk, sp_Wq)  # bs, 1, d*h*w
        # 计算空间注意力分数矩阵
        sp_score = self.sigmoid(sp_Wz.reshape(bs, 1, d, h, w))  # bs, 1, d, h, w
        # 空间增强
        sp_out = sp_score * ch_out

        return sp_out


class LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendInnerProductVector_Simplify(nn.Module):
    """
    使用多尺度感受野信息扩充内积向量长度的局部极化多尺度感受野自注意力模块
    """
    def __init__(self, in_channel, channel, kernels=[1, 3, 5], group=1, dilation=1, r=4):
        """
        定义一个使用多尺度感受野信息扩充内积向量长度的局部极化多尺度感受野自注意力模块

        :param in_channel: 输入通道数
        :param channel: 输出通道数
        :param kernels: 不同分支的内核大小
        :param group: 分组卷积的组数
        :param dilation: 空洞率
        :param r: 衰减率
        """
        super(LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendInnerProductVector_Simplify, self).__init__()
        self.inner_c = channel // (len(kernels) * r)
        # 先定义一个卷积层变换输出通道数
        self.basic_conv = nn.Conv3d(in_channels=in_channel, out_channels=channel, kernel_size=3, padding=1)
        # 定义通道的不同感受野分支的卷积
        self.ch_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, self.inner_c, kernel_size=k, padding=((k - 1) * dilation) // 2, dilation=dilation, groups=group)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])
        # 定义通道注意力的Softmax
        self.ch_softmax = nn.Softmax(1)
        # 定义通道注意力对分数矩阵的卷积Wz
        self.ch_Wz = nn.Conv3d(self.inner_c, channel, kernel_size=1)
        # 定义通道注意力的层归一化
        self.layer_norm = nn.LayerNorm((channel, 1, 1, 1))
        # 定义对注意力分数矩阵的Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

        # 定义空间的不同感受野分支的卷积
        self.sp_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, self.inner_c, kernel_size=k, padding=((k - 1) * dilation) // 2, dilation=dilation, groups=group)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])
        # 定义空间自注意力的Softmax
        self.sp_softmax = nn.Softmax(-1)



    def forward(self, x):
        # 先变换通道数
        x = self.basic_conv(x)
        # 获得输入特征图维度信息
        bs, c, d, h, w = x.size()

        # 计算通道各分支
        ch_outs = [
            conv(x)
            for conv in self.ch_convs
        ]
        # 堆叠通道Wq
        ch_Wq = torch.stack(ch_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 堆叠通道Wk
        ch_Wk = torch.stack(ch_outs, dim=-1).mean(dim=1, keepdim=True)  # bs, 1, d, h, w, k
        # 转换维度
        ch_Wq = ch_Wq.reshape(bs, self.inner_c, -1)  # bs, self.inner_c, d*h*w*k
        ch_Wk = ch_Wk.reshape(bs, -1, 1)  # bs, d*h*w*k, 1
        # 进行Softmax处理
        ch_Wk = self.ch_softmax(ch_Wk)  # bs, d*h*w*k, 1
        # 矩阵相乘
        ch_Wz = torch.matmul(ch_Wq, ch_Wk).unsqueeze(-1).unsqueeze(-1)  # bs, self.inner_c, 1, 1, 1
        # 计算通道注意力分数矩阵
        ch_score = self.sigmoid(self.layer_norm(self.ch_Wz(ch_Wz)))  # bs, c, 1, 1, 1
        # 通道增强
        ch_out = ch_score * x

        # 计算空间Wq各分支
        sp_outs = [
            conv(ch_out)
            for conv in self.sp_convs
        ]
        # 堆叠空间Wq
        sp_Wq = torch.stack(sp_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 堆叠空间Wk
        sp_Wk = torch.stack(sp_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 转换维度
        sp_Wq = sp_Wq.reshape(bs, -1, d * h * w)  # bs, k*self.inner_c, d*h*w
        sp_Wk = sp_Wk.mean(-1).mean(-1).mean(-1).reshape(bs, 1, -1)  # bs, 1, k*self.inner_c
        # 进行Softmax处理
        sp_Wk = self.sp_softmax(sp_Wk)  # bs, 1, k*self.inner_c
        # 矩阵相乘
        sp_Wz = torch.matmul(sp_Wk, sp_Wq)  # bs, 1, d*h*w
        # 计算空间注意力分数矩阵
        sp_score = self.sigmoid(sp_Wz.reshape(bs, 1, d, h, w))  # bs, 1, d, h, w
        # 空间增强
        sp_out = sp_score * ch_out

        return sp_out


class LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendAttentionPoints(nn.Module):
    """
    使用多尺度感受野信息扩充注意力位置的局部极化多尺度感受野自注意力模块
    """
    def __init__(self, in_channel, channel, kernels=[1, 3, 5], group=1, dilation=1, r=4):
        """
        定义一个使用多尺度感受野信息扩充注意力位置的局部极化多尺度感受野自注意力模块

        :param in_channel: 输入通道数
        :param channel: 输出通道数
        :param kernels: 不同分支的内核大小
        :param group: 分组卷积的组数
        :param dilation: 空洞率
        :param r: 衰减率
        """
        super(LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendAttentionPoints, self).__init__()
        self.inner_c = channel // (len(kernels) * r)
        self.k = len(kernels)
        # 先定义一个卷积层变换输出通道数
        self.basic_conv = nn.Conv3d(in_channels=in_channel, out_channels=channel, kernel_size=3, padding=1)
        # 定义通道的不同感受野分支的卷积
        self.ch_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, self.inner_c, kernel_size=k, padding=((k - 1) * dilation) // 2, dilation=dilation, groups=group)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])
        # 定义通道Wv的不同感受野分支的卷积
        self.ch_Wv_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv",
                 nn.Conv3d(channel, self.inner_c, kernel_size=k, padding=((k - 1) * dilation) // 2, dilation=dilation,
                           groups=group)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])
        # 定义通道注意力的Softmax
        self.ch_softmax = nn.Softmax(1)
        # 定义通道注意力对分数矩阵的卷积Wz
        self.ch_Wz = nn.Conv3d(self.k * self.inner_c, self.k * self.inner_c, kernel_size=1)
        # 定义通道注意力的层归一化
        self.layer_norm = nn.LayerNorm((self.k * self.inner_c, 1, 1, 1))
        # 定义对注意力分数矩阵的Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
        # 通道注意力恢复通道数
        self.ch_excit = nn.Conv3d(self.inner_c, channel, kernel_size=1)

        # 定义空间的不同感受野分支的卷积
        self.sp_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, self.inner_c, kernel_size=k, padding=((k - 1) * dilation) // 2, dilation=dilation, groups=group)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])
        # 定义空间Wv的不同感受野分支的卷积
        self.sp_Wv_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv",
                 nn.Conv3d(channel, self.inner_c, kernel_size=k, padding=((k - 1) * dilation) // 2, dilation=dilation,
                           groups=group)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])
        # 定义空间自注意力的Softmax
        self.sp_softmax = nn.Softmax(-1)
        # 空间注意力恢复通道数
        self.sp_excit = nn.Conv3d(self.inner_c, channel, kernel_size=1)



    def forward(self, x):
        # 先变换通道数
        x = self.basic_conv(x)
        # 获得输入特征图维度信息
        bs, c, d, h, w = x.size()


        # 计算通道各分支
        ch_outs = [
            conv(x)
            for conv in self.ch_convs
        ]
        # 堆叠通道Wq
        ch_Wq = torch.stack(ch_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 堆叠通道Wk
        ch_Wk = torch.cat(ch_outs, dim=1).mean(dim=1, keepdim=True)  # bs, 1, d, h, w
        # 计算通道Wv各分支
        ch_Wv_outs = [
            conv(x)
            for conv in self.ch_Wv_convs
        ]
        # 堆叠通道Wv
        ch_Wv = torch.stack(ch_Wv_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 转换维度
        ch_Wq = ch_Wq.reshape(bs, -1, d * h * w)  # bs, k * self.inner_c, d*h*w
        ch_Wk = ch_Wk.reshape(bs, -1, 1)  # bs, d*h*w, 1
        # 进行Softmax处理
        ch_Wk = self.ch_softmax(ch_Wk)  # bs, d*h*w*k, 1
        # 矩阵相乘
        ch_Wz = torch.matmul(ch_Wq, ch_Wk).unsqueeze(-1).unsqueeze(-1)  # bs, k * self.inner_c, 1, 1, 1
        # 计算通道注意力分数矩阵
        ch_score = self.sigmoid(self.layer_norm(self.ch_Wz(ch_Wz))).reshape(bs, -1, self.inner_c, 1, 1, 1)  # bs, k, self.inner_c, 1, 1, 1
        # 通道增强
        ch_out = torch.sum(ch_score * ch_Wv, dim=1)  # bs, self.inner_c, d, h, w
        # 恢复通道数
        ch_out = self.ch_excit(ch_out)  # bs, c, d, h, w

        # 计算空间各分支
        sp_outs = [
            conv(ch_out)
            for conv in self.sp_convs
        ]
        # 堆叠空间Wq
        sp_Wq = torch.stack(sp_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 堆叠空间Wk
        sp_Wk = torch.stack(sp_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 计算空间Wv各分支
        sp_Wv_outs = [
            conv(x)
            for conv in self.sp_Wv_convs
        ]
        # 堆叠通道Wv
        sp_Wv = torch.stack(sp_Wv_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 转换维度
        sp_Wq = sp_Wq.reshape(bs, self.inner_c, -1)  # bs, self.inner_c, d*h*w*k
        sp_Wk = sp_Wk.mean(-1).mean(-1).mean(-1).mean(-1).reshape(bs, 1, -1)  # bs, 1, self.inner_c
        # 进行Softmax处理
        sp_Wk = self.sp_softmax(sp_Wk)  # bs, 1, self.inner_c
        # 矩阵相乘
        sp_Wz = torch.matmul(sp_Wk, sp_Wq)  # bs, 1, d*h*w*k
        # 计算空间注意力分数矩阵
        sp_score = self.sigmoid(sp_Wz.reshape(bs, 1, d, h, w, self.k))  # bs, 1, d, h, w, k
        # 空间增强
        sp_out = torch.sum(sp_score * sp_Wv, dim=-1)  # bs, self.inner_c, d, h, w
        # 恢复通道数
        sp_out = self.sp_excit(sp_out)  # bs, c, d, h, w

        return sp_out


class LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendAttentionPoints_Simplify(nn.Module):
    """
    使用多尺度感受野信息扩充注意力位置的局部极化多尺度感受野自注意力模块
    """
    def __init__(self, in_channel, channel, kernels=[1, 3, 5], group=1, dilation=1, r=4):
        """
        定义一个使用多尺度感受野信息扩充注意力位置的局部极化多尺度感受野自注意力模块

        :param in_channel: 输入通道数
        :param channel: 输出通道数
        :param kernels: 不同分支的内核大小
        :param group: 分组卷积的组数
        :param dilation: 空洞率
        :param r: 衰减率
        """
        super(LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendAttentionPoints_Simplify, self).__init__()
        self.inner_c = channel // (len(kernels) * r)
        self.k = len(kernels)
        # 先定义一个卷积层变换输出通道数
        self.basic_conv = nn.Conv3d(in_channels=in_channel, out_channels=channel, kernel_size=3, padding=1)
        # 定义通道的不同感受野分支的卷积
        self.ch_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, self.inner_c, kernel_size=k, padding=((k - 1) * dilation) // 2, dilation=dilation, groups=group)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])
        # 定义通道注意力的Softmax
        self.ch_softmax = nn.Softmax(1)
        # 定义通道注意力对分数矩阵的卷积Wz
        self.ch_Wz = nn.Conv3d(self.k * self.inner_c, self.k * self.inner_c, kernel_size=1)
        # 定义通道注意力的层归一化
        self.layer_norm = nn.LayerNorm((self.k * self.inner_c, 1, 1, 1))
        # 定义对注意力分数矩阵的Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()
        # 通道注意力恢复通道数
        self.ch_excit = nn.Conv3d(self.inner_c, channel, kernel_size=1)

        # 定义空间的不同感受野分支的卷积
        self.sp_convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, self.inner_c, kernel_size=k, padding=((k - 1) * dilation) // 2, dilation=dilation, groups=group)),
                ("bn", nn.BatchNorm3d(self.inner_c)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])
        # 定义空间自注意力的Softmax
        self.sp_softmax = nn.Softmax(-1)
        # 空间注意力恢复通道数
        self.sp_excit = nn.Conv3d(self.inner_c, channel, kernel_size=1)



    def forward(self, x):
        # 先变换通道数
        x = self.basic_conv(x)
        # 获得输入特征图维度信息
        bs, c, d, h, w = x.size()

        # 计算通道各分支
        ch_outs = [
            conv(x)
            for conv in self.ch_convs
        ]
        # 堆叠通道Wq
        ch_Wq = torch.stack(ch_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 堆叠通道Wk
        ch_Wk = torch.cat(ch_outs, dim=1).mean(dim=1, keepdim=True)  # bs, 1, d, h, w
        # 堆叠通道Wv
        ch_Wv = torch.stack(ch_outs, dim=1)  # bs, k, self.inner_c, d, h, w
        # 转换维度
        ch_Wq = ch_Wq.reshape(bs, -1, d * h * w)  # bs, k * self.inner_c, d*h*w
        ch_Wk = ch_Wk.reshape(bs, -1, 1)  # bs, d*h*w, 1
        # 进行Softmax处理
        ch_Wk = self.ch_softmax(ch_Wk)  # bs, d*h*w*k, 1
        # 矩阵相乘
        ch_Wz = torch.matmul(ch_Wq, ch_Wk).unsqueeze(-1).unsqueeze(-1)  # bs, k * self.inner_c, 1, 1, 1
        # 计算通道注意力分数矩阵
        ch_score = self.sigmoid(self.layer_norm(self.ch_Wz(ch_Wz))).reshape(bs, -1, self.inner_c, 1, 1, 1)  # bs, k, self.inner_c, 1, 1, 1
        # 通道增强
        ch_out = torch.sum(ch_score * ch_Wv, dim=1)  # bs, self.inner_c, d, h, w
        # 恢复通道数
        ch_out = self.ch_excit(ch_out)  # bs, c, d, h, w

        # 计算空间各分支
        sp_outs = [
            conv(ch_out)
            for conv in self.sp_convs
        ]
        # 堆叠空间Wq
        sp_Wq = torch.stack(sp_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 堆叠空间Wk
        sp_Wk = torch.stack(sp_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 堆叠空间Wv
        sp_Wv = torch.stack(sp_outs, dim=-1)  # bs, self.inner_c, d, h, w, k
        # 转换维度
        sp_Wq = sp_Wq.reshape(bs, self.inner_c, -1)  # bs, self.inner_c, d*h*w*k
        sp_Wk = sp_Wk.mean(-1).mean(-1).mean(-1).mean(-1).reshape(bs, 1, -1)  # bs, 1, self.inner_c
        # 进行Softmax处理
        sp_Wk = self.sp_softmax(sp_Wk)  # bs, 1, self.inner_c
        # 矩阵相乘
        sp_Wz = torch.matmul(sp_Wk, sp_Wq)  # bs, 1, d*h*w*k
        # 计算空间注意力分数矩阵
        sp_score = self.sigmoid(sp_Wz.reshape(bs, 1, d, h, w, self.k))  # bs, 1, d, h, w, k
        # 空间增强
        sp_out = torch.sum(sp_score * sp_Wv, dim=-1)  # bs, self.inner_c, d, h, w
        # 恢复通道数
        sp_out = self.sp_excit(sp_out)  # bs, c, d, h, w

        return sp_out



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = torch.randn((4, 32, 64, 64, 64)).to(device)

    # model = LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendChannel(channel=32).to(device)
    # model = LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_SelectiveKernel_Parallel(32).to(device)
    # model = LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendInnerProductVector(32, 64).to(device)
    # model = LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendInnerProductVector_Simplify(32, 64).to(device)
    model = LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendAttentionPoints(32, 64).to(device)

    output = model(x)

    print(x.size())
    print(output.size())
















