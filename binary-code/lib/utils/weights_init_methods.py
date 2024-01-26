# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/11/14 15:25
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch.nn as nn



def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



def weights_init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError("初始化方法：[%s]，没有实现！" % init_type)