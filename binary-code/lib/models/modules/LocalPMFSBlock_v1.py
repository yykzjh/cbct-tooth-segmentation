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




class LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendChannel(nn.Module):
    """
    简单使用多尺度感受野信息扩充通道维度的局部极化多尺度感受野自注意力模块
    """
    def __init__(self, channel=512, kernels=[1, 3, 5], group=1, dilation=1):
        """
        定义一个简单使用多尺度感受野信息扩充通道维度的局部极化多尺度感受野自注意力模块

        :param channel: 输入通道数
        :param kernels: 不同分支的内核大小
        :param group: 分组卷积的组数
        :param dilation: 空洞率
        """
        super(LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_ExtendChannel, self).__init__()
        # 定义不同内核大小的分支
        self.convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, channel, kernel_size=k, padding=((k - 1) * dilation) // 2, dilation=dilation, groups=group)),
                ("bn", nn.BatchNorm3d(channel)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])
        # 定义一个串行极化自注意力(PSA)模块
        self.psa = SequentialPolarizedSelfAttention3d(len(kernels) * channel)
        # 定义一个卷积层将通道数变为原样
        self.conv1x1 = nn.Conv3d(len(kernels) * channel, channel, kernel_size=1)


    def forward(self, x):
        # 获得输入特征图维度信息
        bs, c, d, h, w = x.size()

        # 计算各分支不同内核大小的卷积结果
        conv_outs = [
            conv(x)
            for conv in self.convs
        ]

        # 将不同分支的结果在通道维度cat
        cat_out = torch.cat(conv_outs, dim=1)
        # 经过psa处理
        psa_out = self.psa(cat_out)
        # 降通道数
        out = self.conv1x1(psa_out)

        return out



class PCSA_Score(nn.Module):
    """
    极化通道自注意力主干分数模块
    """
    def __init__(self, channel, inner_d):
        """
        定义一个极化通道自注意力主干分数模块

        :param channel: 输入通道数
        :param inner_d: 内部和输出通道数
        """
        super(PCSA_Score, self).__init__()
        # 定义通道自注意的两个1X1卷积Wv、Wq
        self.ch_Wv = nn.Conv3d(channel, inner_d, kernel_size=1)
        self.ch_Wq = nn.Conv3d(channel, 1, kernel_size=1)
        # 定义通道注意力的Softmax
        self.ch_softmax = nn.Softmax(1)
        # 定义通道注意力对分数矩阵的卷积Wz
        self.ch_Wz = nn.Conv3d(inner_d, inner_d, kernel_size=1)
        # 定义通道注意力的层归一化
        self.layer_norm = nn.LayerNorm((inner_d, 1, 1, 1))
        # 定义对注意力分数矩阵的Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # 获取输入特征图张量的维度信息
        bs, c, d, h, w = x.size()

        # Channel-only Self-Attention
        ch_Wv = self.ch_Wv(x)  # bs,inner_d,d,h,w
        ch_Wq = self.ch_Wq(x)  # bs,1,d,h,w
        ch_Wv = ch_Wv.reshape(bs, -1, d*h*w)  # bs,inner_d,d*h*w
        ch_Wq = ch_Wq.reshape(bs, -1, 1)  # bs,d*h*w,1
        ch_Wq = self.ch_softmax(ch_Wq)  # bs,d*h*w,1
        ch_Wz = torch.matmul(ch_Wv, ch_Wq).unsqueeze(-1).unsqueeze(-1)  # bs,inner_d,1,1,1
        ch_score = self.sigmoid(self.layer_norm(self.ch_Wz(ch_Wz)))  # bs,inner_d,1,1,1
        ch_score = torch.squeeze(ch_score)

        return ch_score



class PSSA_Score(nn.Module):
    """
    极化空间自注意力主干分数模块
    """
    def __init__(self, channel):
        """
        定义一个极化空间自注意力主干分数模块

        :param channel: 输入通道数
        """
        super(PSSA_Score, self).__init__()
        # 定义空间自注意的两个1X1卷积Wv、Wq
        self.sp_Wv = nn.Conv3d(channel, channel // 2, kernel_size=1)
        self.sp_Wq = nn.Conv3d(channel, channel // 2, kernel_size=1)
        # 定义空间自注意力的全局自适应平均池化
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # 定义空间自注意力的Softmax
        self.sp_softmax = nn.Softmax(-1)
        # 定义对注意力分数矩阵的Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # 获取输入特征图张量的维度信息
        bs, c, d, h, w = x.size()

        sp_Wv = self.sp_Wv(x)  # bs,c//2,d,h,w
        sp_Wq = self.sp_Wq(x)  # bs,c//2,d,h,w
        sp_Wq = self.avg_pool(sp_Wq)  # bs,c//2,1,1,1
        sp_Wv = sp_Wv.reshape(bs, c // 2, -1)  # bs,c//2,d*h*w
        sp_Wq = sp_Wq.reshape(bs, 1, c // 2)  # bs,1,c//2
        sp_Wq = self.sp_softmax(sp_Wq)  # bs,1,c//2
        sp_Wz = torch.matmul(sp_Wq, sp_Wv)  # bs,1,d*h*w
        sp_score = self.sigmoid(sp_Wz.reshape(bs, 1, d, h, w))  # bs, 1, d, h, w

        return sp_score



class LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_SelectiveKernel_Parallel(nn.Module):
    """
    通过极化自注意力模块计算选择性核的主干分数的局部极化多尺度感受野自注意力模块
    """
    def __init__(self, channel, kernels=[1, 3, 5], r=16, group=1, L=32, dilation=1):
        """
        定义一个通过极化自注意力模块计算选择性核的主干分数的局部极化多尺度感受野自注意力模块

        :param channel: 输入通道数
        :param kernels: 不同分支的内核大小
        :param r: 通道数衰减系数
        :param group: 分组卷积的组数
        :param L: 通道数衰减最小值
        :param dilation: 空洞率
        """
        super(LocalPolarizedMultiScaleReceptiveFieldSelfAttentionBlock_SelectiveKernel_Parallel, self).__init__()
        # 定义不同内核大小的分支
        self.convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("conv", nn.Conv3d(channel, channel, kernel_size=k, padding=((k - 1) * dilation) // 2, dilation=dilation, groups=group)),
                ("bn", nn.BatchNorm3d(channel)),
                ("relu", nn.ReLU(inplace=True))
            ]))
            for k in kernels
        ])

        self.ch_in = max(channel // r, L)
        self.ch_PSA_score = PCSA_Score(channel=channel, inner_d=self.ch_in)
        # 定义各分支的通道注意力描述符
        self.ch_fcs = nn.ModuleList([
            nn.Linear(self.ch_in, channel)
            for i in range(len(kernels))
        ])

        self.sp_PSA_score = PSSA_Score(channel=channel)
        # 定义各分支的通道注意力描述符
        self.sp_convs = nn.ModuleList([
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=9, padding=4)
            for i in range(len(kernels))
        ])

        # 定义应用于不同分支的Softmax
        self.softmax = nn.Softmax(dim=0)


    def forward(self, x):
        # 获得输入特征图维度信息
        bs, c, d, h, w = x.size()

        # 计算各分支不同内核大小的卷积结果
        conv_outs = [
            conv(x)
            for conv in self.convs
        ]
        # 将各分支得到的特征图逐元素相加
        U = sum(conv_outs)  # bs, c, d, h, w
        # 将之前不同分支卷积得到的结果堆叠起来
        features_stack = torch.stack(conv_outs, dim=0)  # k, bs, c, d, h, w

        # 计算通道主干分数
        ch_z = self.ch_PSA_score(U)
        # 根据通道主干分数计算各分支的通道注意力描述符
        ch_weights = [
            fc(ch_z).view(bs, c, 1, 1, 1)
            for fc in self.ch_fcs
        ]
        # 堆叠各分支的通道注意力描述符
        ch_weights_stack = torch.stack(ch_weights, 0)  # k, bs, c, 1, 1, 1
        # 对同一个通道不同分支的注意力数值进行Softmax
        ch_weights_stack = self.softmax(ch_weights_stack)  # k, bs, c, 1, 1, 1
        # 根据注意力分数融合各分支的特征图
        ch_out = torch.sum(ch_weights_stack * features_stack, dim=0)  # bs, c, d, h, w

        # 计算空间主干分数
        sp_z = self.sp_PSA_score(U)
        # 根据空间主干分数计算各分支的空间注意力描述符
        sp_weights = [
            conv(sp_z)
            for conv in self.sp_convs
        ]
        # 堆叠各分支的通道注意力描述符
        sp_weights_stack = torch.stack(sp_weights, 0)  # k, bs, 1, d, h, w
        # 对同一个通道不同分支的注意力数值进行Softmax
        sp_weights_stack = self.softmax(sp_weights_stack)  # k, bs, 1, d, h, w
        # 根据注意力分数融合各分支的特征图
        sp_out = torch.sum(sp_weights_stack * features_stack, dim=0)  # bs, c, d, h, w

        out = ch_out + sp_out

        return out





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
















