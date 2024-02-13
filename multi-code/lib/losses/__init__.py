# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/1 22:50
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
from .DiceLoss import *



def get_loss_function(opt):
    if opt["loss_function_name"] == "DiceLoss":
        loss_function = DiceLoss(opt["classes"], weight=torch.FloatTensor(opt["class_weight"]).to(opt["device"]), sigmoid_normalization=False, mode=opt["dice_loss_mode"])

    elif opt["loss_function_name"] == "BCEWithLogitsLoss":
        loss_function = nn.BCEWithLogitsLoss()

    else:
        raise RuntimeError(f"{opt['loss_function_name']}是不支持的损失函数！")

    return loss_function


