# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/4/19 15:17
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
from torch.utils.data import DataLoader

from .BinaryToothDataset import BinaryToothDataset
from .BinaryToothSurfaceDataset import BinaryToothSurfaceDataset


def get_dataloader(opt):
    """
    获取数据加载器
    Args:
        opt: 参数字典
    Returns:
    """
    if opt["dataset_name"] == "BINARY-TOOTH-FULL":
        # 初始化数据集
        train_set = BinaryToothDataset(opt, mode="train")
        valid_set = BinaryToothDataset(opt, mode="valid")

    elif opt["dataset_name"] == "BINARY-TOOTH-SURFACE":
        # 初始化数据集
        train_set = BinaryToothSurfaceDataset(opt, mode="train")
        valid_set = BinaryToothSurfaceDataset(opt, mode="valid")

    else:
        raise RuntimeError(f"{opt['dataset_name']}是不支持的数据集！")

    # 初始化数据加载器
    train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # 存储steps_per_epoch
    opt["steps_per_epoch"] = len(train_set) // opt["batch_size"]

    return train_loader, valid_loader


def get_test_dataloader(opt):
    """
    获取测试集数据加载器
    :param opt: 参数字典
    :return:
    """
    if opt["dataset_name"] == "BINARY-TOOTH-FULL":
        # 初始化数据集
        valid_set = BinaryToothDataset(opt, mode="valid")

    elif opt["dataset_name"] == "BINARY-TOOTH-SURFACE":
        # 初始化数据集
        valid_set = BinaryToothSurfaceDataset(opt, mode="valid")

    else:
        raise RuntimeError(f"{opt['dataset_name']}是不支持的数据集！")

    # 初始化数据加载器
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    return valid_loader
