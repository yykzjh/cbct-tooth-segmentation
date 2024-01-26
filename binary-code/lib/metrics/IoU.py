# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/6/13 2:54
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch
import torch.nn as nn



class IoU(object):
    def __init__(self, num_classes=33, sigmoid_normalization=False):
        """
        定义IoU评价指标计算器

        :param num_classes: 类别数
        :param sigmoid_normalization: 对网络输出采用sigmoid归一化方法，否则采用softmax
        """
        super(IoU, self).__init__()
        # 初始化参数
        self.num_classes = num_classes
        # 初始化sigmoid或者softmax归一化方法
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    @staticmethod
    def compute_per_channel_iou(pred, label, num_classes):
        """
        计算各类别的IoU

        :param pred: 预测标签图
        :param label: 真实标签图
        :param num_classes: 通道和类别数
        :return:
        """
        # 求所有类别的交集
        intersect = pred[pred == label]
        # 统计各类别交集点数量
        intersect_count = torch.histc(
            intersect.float(), bins=num_classes, min=0, max=num_classes - 1)
        # 统计预测标签图中各类别点数量
        pred_count = torch.histc(
            pred.float(), bins=num_classes, min=0, max=num_classes - 1)
        # 统计真实标签图中各类别点数量
        label_count = torch.histc(
            label.float(), bins=num_classes, min=0, max=num_classes - 1)
        # 根据指标类型计算分母
        union_count = pred_count + label_count - intersect_count
        # 计算每个通道的iou
        per_channel_iou = intersect_count / (union_count + 1e-12)
        # 将没有出现的类别的值置为-1.0
        for i in range(num_classes):
            if label_count[i] == 0:
                per_channel_iou[i] = -1.0

        return per_channel_iou

    def __call__(self, input, target):
        """
        IoU

        :param input: 网络模型输出的预测图,(B, C, H, W, D)
        :param target: 标注图像,(B, H, W, D)
        :return:
        """
        # 对预测图进行Sigmiod或者Sofmax归一化操作
        input = self.normalization(input)

        # 将预测图像进行分割
        seg = torch.argmax(input, dim=1)
        # 判断预测图和真是标签图的维度大小是否一致
        assert seg.shape == target.shape, "seg和target的维度大小不一致"

        # 转换seg和target数据类型为布尔型
        seg = seg.float()
        target = target.float()

        # 判断标注图和分割图的维度是否都是4维
        assert seg.dim() == target.dim() == 4, "IoU: 标注图和分割图的维度不是都为4维！"
        # 判断标注图和分割图的尺寸是否一致
        assert seg.size() == target.size(), "IoU: 分割图和标注图的尺寸不一致！"

        # 计算每个通道上的iou
        per_channel_iou = self.compute_per_channel_iou(seg, target, self.num_classes)
        # 获取有效通道的mask
        mask = torch.ones((self.num_classes,))
        for i in range(self.num_classes):
            if per_channel_iou[i] == -1.0:
                mask[i] = 0.0
        # 返回有效通道的平均iou
        return (torch.sum(per_channel_iou * mask) / torch.sum(mask)).item()
