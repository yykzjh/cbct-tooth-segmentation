import time
import torch
import torch.nn as nn
import surface_distance as sd

from lib.utils import expand_as_one_hot


class HausdorffDistance(object):
    def __init__(self, num_classes=33, sigmoid_normalization=False):
        """
        定义豪斯多夫距离评价指标计算器

        :param num_classes: 类别数
        :param sigmoid_normalization: 对网络输出采用sigmoid归一化方法，否则采用softmax
        """
        super(HausdorffDistance, self).__init__()
        # 初始化参数
        self.num_classes = num_classes
        # 初始化sigmoid或者softmax归一化方法
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    @staticmethod
    def compute_per_channel_hd(seg, target, num_classes):
        """
        计算各类别的hd
        :param seg: 预测分割后的分割图
        :param target: 真实标签图
        :param num_classes: 通道和类别数
        :return:
        """
        # 获取各维度大小
        bs, _, h, w, d = seg.shape
        # 初始化输出
        output = torch.full((bs, num_classes), -1.0)

        # 计算HD
        for b in range(bs):  # 遍历batch
            # 遍历各类别
            for cla in range(num_classes):
                # 分别计算两个表面点集合中各点到对面集合的距离
                surface_distances = sd.compute_surface_distances(target[b, cla, ...].numpy(), seg[b, cla, ...].numpy(),
                                                                 spacing_mm=(1.0, 1.0, 1.0))

                if len(surface_distances["distances_pred_to_gt"]) == 0 or len(surface_distances["distances_gt_to_pred"]) == 0:
                    continue
                # 计算一张图像一个类别的hd
                output[b, cla] = sd.compute_robust_hausdorff(surface_distances, 95)
        # 计算各类别在batch上的平均HD
        out = torch.full((num_classes,), -1.0)
        for cla in range(num_classes):
            cnt = 0
            acc_sum = 0
            for b in range(bs):
                if output[b, cla] != -1.0:
                    acc_sum += output[b, cla]
                    cnt += 1.0
            if cnt > 0:
                out[cla] = acc_sum / cnt
        return out

    def __call__(self, input, target):
        """
        计算豪斯多夫距离评价指标

        :param input: 网络模型输出的预测图,(B, C, H, W, D)
        :param target: 标注图像,(B, H, W, D)
        :return: 各batch中各类别牙齿的豪斯多夫距离
        """
        # 对预测图进行Sigmiod或者Sofmax归一化操作
        input = self.normalization(input)

        # 将预测图像进行分割
        seg = torch.argmax(input, dim=1)
        # 判断预测图和真是标签图的维度大小是否一致
        assert seg.shape == target.shape, "seg和target的维度大小不一致"

        # 转换seg和target数据类型为整型
        seg = seg.long()
        target = target.long()

        # 将分割图和标签图都进行one-hot处理
        seg = expand_as_one_hot(seg, self.num_classes)
        target = expand_as_one_hot(target, self.num_classes)

        # 转换seg和target数据类型为布尔型
        seg = seg.bool()
        target = target.bool()

        # 判断one-hot处理后标注图和分割图的维度是否都是5维
        assert seg.dim() == target.dim() == 5, "HD: one-hot处理后标注图和分割图的维度不是都为5维！"
        # 判断one-hot处理后标注图和分割图的尺寸是否一致
        assert seg.size() == target.size(), "HD: one-hot处理后分割图和标注图的尺寸不一致！"

        # 计算每个通道上的hd
        per_channel_hd = self.compute_per_channel_hd(seg, target, self.num_classes)
        # 获取有效通道的mask
        mask = torch.ones((self.num_classes, ))
        for i in range(self.num_classes):
            if per_channel_hd[i] == -1.0:
                mask[i] = 0.0
        # 返回有效通道的平均hd
        return (torch.sum(per_channel_hd * mask) / torch.sum(mask)).item()




