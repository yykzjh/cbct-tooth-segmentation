# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/02/14 02:52
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import torch
import torch.nn as nn

from .PMFSNet import PMFSNet
import lib.utils as utils


class TwoStageNet(nn.Module):

    def __init__(self, opt, stage2_model, in_channels=1):
        """
        初始化两阶段分割架构

        :param opt: 参数字典
        :param stage2_model: 第二阶段的模型
        :param in_channels: 输入特征图通道数
        """
        super(TwoStageNet, self).__init__()
        # 初始化参数
        self.opt = opt
        self.stage2_model = stage2_model
        self.in_channels = in_channels

        # 初始化第一阶段的模型
        self.stage1_surface_model = PMFSNet(in_channels=self.in_channels, out_channels=2, dim="3d", scaling_version="TINY", with_pmfs_block=False).to(self.opt["device"])

        # 加载第一阶段模型参数
        self.load()

        # 冻结第一阶段模型参数
        self.stage1_surface_model.requires_grad_(False)

    def forward(self, x):
        # 一阶段
        surface_pred = self.stage1_surface_model(x)

        # 组合一阶段预测结果和原始输入
        surface_logits = nn.Softmax(dim=1)(surface_pred)
        surface_x = torch.argmax(surface_logits, dim=1, keepdim=True)
        merge_x = torch.cat([surface_x, x], dim=1)

        # 二阶段分割
        return self.stage2_model(merge_x)

    def load(self):
        if self.opt["surface_pretrain"] is None:
            print("一阶段表面轮廓预测模型的权重为None")
            utils.init_weights(self.stage1_surface_model, init_type="kaiming")
        else:
            # 加载模型参数字典
            pretrain_state_dict = torch.load(self.opt["surface_pretrain"], map_location=lambda storage, loc: storage.cuda(self.opt["device"]))
            # 获取模型参数字典
            model_state_dict = self.stage1_surface_model.state_dict()
            # 遍历模型参数
            load_count = 0  # 成功加载参数计数
            for param_name in model_state_dict.keys():
                # 判断当前模型参数是否在预训练参数中
                if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                    model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                    load_count += 1
            # 严格加载模型参数
            self.stage1_surface_model.load_state_dict(model_state_dict, strict=True)
            # 输出权重参数加载率
            print("{:.2f}%的一阶段表面轮廓预测模型参数成功加载预训练权重".format(100 * load_count / len(model_state_dict)))
