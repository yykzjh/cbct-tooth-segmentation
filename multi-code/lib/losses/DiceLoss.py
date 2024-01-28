import torch
from torch import nn as nn

from lib.utils import *


class DiceLoss(nn.Module):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    """

    def __init__(self, classes=1, weight=None, sigmoid_normalization=False, mode="extension"):
        super(DiceLoss, self).__init__()
        self.classes = classes
        self.weight = weight
        self.mode = mode
        
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def dice(self, input, target):
        """
        计算网络模型输出的预测图和标注图的Dice系数,每个通道(类别)都算出一个值

        Args:
            input: 网络模型输出的预测图
            target: 标注图

        Returns:

        """
        # ont-hot处理，将标注图在axis=1维度上扩张，该维度大小等于预测图的通道C大小，维度上每一个索引依次对应一个类别,(B, C, H, W, D)
        target = expand_as_one_hot(target.long(), self.classes)

        # 判断one-hot处理后标注图和预测图的维度是否都是5维
        assert input.dim() == target.dim() == 5, "DiceLoss: one-hot处理后标注图和预测图的维度不是都为5维！"
        # 判断one-hot处理后标注图和预测图的尺寸是否一致
        assert input.size() == target.size(), "DiceLoss: one-hot处理后预测图和标注图的尺寸不一致！"

        # 对预测图进行Sigmiod或者Sofmax归一化操作
        input = self.normalization(input)

        return compute_per_channel_dice(input, target, epsilon=1e-12, mode=self.mode)


    def forward(self, input, target):
        """
        计算Dice Loss

        Args:
            input: 网络模型输出的预测图,(B, C, H, W, D)
            target: 标注图像,(B, H, W, D)

        Returns:

        """
        # 计算每个通道(类别)的Dice系数,输出为(C, )
        per_channel_dice = self.dice(input, target)

        # 根据计算出的各通道(类别)是否为0，得到当前batch实际出现的类别mask
        real_weight = self.weight.clone()
        for i, dice in enumerate(per_channel_dice):
            if dice == 0:
                real_weight[i] = 0

        # 对每个类别进行加权求和
        weighted_dsc = torch.sum(per_channel_dice * real_weight) / torch.sum(real_weight)

        # 计算加权后的损失值
        loss = 1. - weighted_dsc

        return loss
