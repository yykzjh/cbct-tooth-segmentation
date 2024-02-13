# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/5/1 18:36
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch
import numpy as np
import torch.nn as nn


class AveragePointToPointErrors(object):
    def __init__(self, num_classes=33, sigmoid_normalization=True):
        """
        定义平均点对点误差(Average Point-to-point Errors,APE)
        Args:
            num_classes: 类别数
            sigmoid_normalization: 对网络输出采用sigmoid归一化方法，否则采用softmax
        """
        super(AveragePointToPointErrors, self).__init__()
        # 初始化参数
        self.num_classes = num_classes
        # 初始化sigmoid或者softmax归一化方法
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    @staticmethod
    def compute_per_channel_pel(pred, label, num_classes):
        """
        计算各关键点的点对点误差(Point-to-point Error for Landmark,PEL)

        :param pred: 预测热力图
        :param label: 真实热力图
        :param num_classes: 通道和类别数
        :return:
        """
        # 获取各维度大小
        bs, _, h, w, d = pred.shape
        # 初始化输出
        output = torch.full((bs, num_classes), -1.0)

        # 计算pel
        for b in range(bs):  # 遍历batch
            # 遍历各类别
            for cla in range(num_classes):
                if torch.max(label[b, cla, ...]) == 0.0:
                    continue
                pred_condidate_point_positions = torch.where(pred[b, cla, ...] == pred[b, cla, ...].max())
                label_condidate_point_positions = torch.where(label[b, cla, ...] == label[b, cla, ...].max())
                pred_x, pred_y, pred_z = pred_condidate_point_positions[0][0].item() / (h - 1), pred_condidate_point_positions[1][0].item() / (w - 1), \
                                         pred_condidate_point_positions[2][0].item() / (d - 1)
                label_x, label_y, label_z = label_condidate_point_positions[0][0].item() / (h - 1), label_condidate_point_positions[1][0].item() / (w - 1), \
                                            label_condidate_point_positions[2][0].item() / (d - 1)
                d = np.sqrt((pred_x - label_x) ** 2 + (pred_y - label_y) ** 2 + (pred_z - label_z) ** 2)
                # 计算一张图像一个类别的pel
                output[b, cla] = d
        # 计算各类别在batch上的pel
        out = torch.full((num_classes,), -1.0)
        for cla in range(num_classes):
            cnt = 0
            pel_sum = 0
            for b in range(bs):
                if output[b, cla] != -1.0:
                    pel_sum += output[b, cla]
                    cnt += 1.0
            if cnt > 0:
                out[cla] = pel_sum / cnt
        return out

    def __call__(self, input, target):
        """
        计算平均点对点误差(Average Point-to-point Errors,APE)

        :param input: 网络模型输出的预测图,(B, C, H, W, D)
        :param target: 标注热力图,(B, C, H, W, D)
        :return:
        """
        # 对预测图进行Sigmiod或者Sofmax归一化操作
        input = self.normalization(input)

        # 判断input和target尺寸是否一致
        assert input.size() == target.size()
        assert input.size(1) == self.num_classes

        # 计算各关键点的点对点误差PEL
        per_channel_pel = self.compute_per_channel_pel(input, target, self.num_classes)
        # 获取有效通道的mask
        mask = torch.ones((self.num_classes,))
        for i in range(self.num_classes):
            if per_channel_pel[i] == -1.0:
                mask[i] = 0.0
        # 返回有效通道的平均点对点误差APE
        return (torch.sum(per_channel_pel * mask) / torch.sum(mask)).item()
