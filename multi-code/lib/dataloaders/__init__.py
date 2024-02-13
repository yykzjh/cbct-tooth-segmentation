# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/4/19 15:17
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
from torch.utils.data import DataLoader

from .MultipleToothDataset import MultipleToothDataset
from .MultipleToothSurfaceDataset import MultipleToothSurfaceDataset
from .MultipleToothCentroidDataset import MultipleToothCentroidDataset


def get_dataloader(opt, train_images_path_list=None, train_labels_path_list=None, valid_images_path_list=None, valid_labels_path_list=None):
    """
    获取数据加载器
    Args:
        opt: 参数字典
        train_images_path_list: 训练集所有原始图像路径
        train_labels_path_list: 训练集所有标签图像路径
        valid_images_path_list: 验证集所有原始图像路径
        valid_labels_path_list: 验证集所有标签图像路径
    Returns:
    """
    if opt["dataset_name"] == "MULTIPLE-TOOTH":
        # 初始化数据集
        train_set = MultipleToothDataset(opt, train_images_path_list, train_labels_path_list, mode="train")
        valid_set = MultipleToothDataset(opt, valid_images_path_list, valid_labels_path_list, mode="valid")

    elif opt["dataset_name"] == "MULTIPLE-TOOTH-SURFACE":
        # 初始化数据集
        train_set = MultipleToothSurfaceDataset(opt, mode="train")
        valid_set = MultipleToothSurfaceDataset(opt, mode="valid")

    elif opt["dataset_name"] == "MULTIPLE-TOOTH-CENTROID":
        # 初始化数据集
        train_set = MultipleToothCentroidDataset(opt, mode="train")
        valid_set = MultipleToothCentroidDataset(opt, mode="valid")

    else:
        raise RuntimeError(f"{opt['dataset_name']}是不支持的数据集！")

    # 初始化数据加载器
    train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # 存储steps_per_epoch
    opt["steps_per_epoch"] = len(train_set) // opt["batch_size"]

    return train_loader, valid_loader



