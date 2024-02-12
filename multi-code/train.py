# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/1 22:52
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import gc
import glob
import math
import tqdm
import shutil
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from lib import utils, dataloaders, models, losses, metrics, trainers

# 默认参数,这里的参数在后面添加到模型中，以params['dropout_rate']等替换原来的参数
params = {
    # ——————————————————————————————————————————————     启动初始化    ———————————————————————————————————————————————————

    "CUDA_VISIBLE_DEVICES": "0",  # 选择可用的GPU编号

    "seed": 20240128,  # 随机种子

    "cuda": True,  # 是否使用GPU

    "benchmark": False,  # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。用于网络输入维度和类型不会变的情况。

    "deterministic": True,  # 固定cuda的随机数种子，每次返回的卷积算法将是确定的。用于复现模型结果

    # —————————————————————————————————————————————     预处理       ————————————————————————————————————————————————————

    "resample_spacing": [0.5, 0.5, 0.5],  # 重采样的体素间距。三个维度的值一般相等，可设为0.5(图像尺寸有[200,200,100]、[200,200,200]、
    # [160,160,160]),或者设为0.25(图像尺寸有[400,400,200]、[400,400,400]、[320,320,320])

    "clip_lower_bound": -3556,  # clip的下边界数值
    "clip_upper_bound": 12419,  # clip的上边界数值

    "samples_train": 256,  # 作为实际的训练集采样的子卷数量，也就是在原训练集上随机裁剪的子图像数量

    "crop_size": (160, 160, 96),  # 随机裁剪的尺寸。1、每个维度都是32的倍数这样在下采样时不会报错;2、11G的显存最大尺寸不能超过(192,192,160);
    # 3、要依据上面设置的"resample_spacing",在每个维度随机裁剪的尺寸不能超过图像重采样后的尺寸;

    "crop_threshold": 0.5,  # 随机裁剪时需要满足的条件，不满足则重新随机裁剪的位置。条件表示的是裁剪区域中的前景占原图总前景的最小比例

    # ——————————————————————————————————————————————    数据增强    ——————————————————————————————————————————————————————

    "augmentation_probability": 0.8,  # 每张图像做数据增强的概率
    "augmentation_method": "Compose",  # 数据增强的方式，可选["Compose", "Choice"]

    # 弹性形变参数
    "open_elastic_transform": True,  # 是否开启弹性形变数据增强
    "elastic_transform_sigma": 20,  # 高斯滤波的σ,值越大，弹性形变越平滑
    "elastic_transform_alpha": 1,  # 形变的幅度，值越大，弹性形变越剧烈

    # 高斯噪声参数
    "open_gaussian_noise": True,  # 是否开启添加高斯噪声数据增强
    "gaussian_noise_mean": 0,  # 高斯噪声分布的均值
    "gaussian_noise_std": 0.01,  # 高斯噪声分布的标准差,值越大噪声越强

    # 随机翻转参数
    "open_random_flip": True,  # 是否开启随机翻转数据增强

    # 随机缩放参数
    "open_random_rescale": True,  # 是否开启随机缩放数据增强
    "random_rescale_min_percentage": 0.5,  # 随机缩放时,最小的缩小比例
    "random_rescale_max_percentage": 1.5,  # 随机缩放时,最大的放大比例

    # 随机旋转参数
    "open_random_rotate": True,  # 是否开启随机旋转数据增强
    "random_rotate_min_angle": -50,  # 随机旋转时,反方向旋转的最大角度
    "random_rotate_max_angle": 50,  # 随机旋转时,正方向旋转的最大角度

    # 随机位移参数
    "open_random_shift": True,  # 是否开启随机位移数据增强
    "random_shift_max_percentage": 0.3,  # 在图像的三个维度(D,H,W)都进行随机位移，位移量的范围为(-0.3×(D、H、W),0.3×(D、H、W))

    # 标准化均值
    "normalize_mean": 0.2003765726308436,
    "normalize_std": 0.062439472842853866,

    # —————————————————————————————————————————————    数据读取     ——————————————————————————————————————————————————————

    "dataset_name": "MULTIPLE-TOOTH",  # 数据集名称， 可选["MULTIPLE-TOOTH"]

    "dataset_path": r"./datasets/HX-multi-class-10",  # 数据集路径

    "create_data": False,  # 是否重新分割子卷训练集

    "batch_size": 1,  # batch_size大小

    "num_workers": 2,  # num_workers大小

    # —————————————————————————————————————————————    网络模型     ——————————————————————————————————————————————————————

    "model_name": "PMFSNet",  # 模型名称，可选["DenseVNet","UNet3D", "VNet", "AttentionUNet3D", "R2UNet", "R2AttentionUNet",
    # "HighResNet3D", "DenseVoxelNet", "MultiResUNet3D", "DenseASPPUNet", "PMFSNet", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet"]

    "in_channels": 1,  # 模型最开始输入的通道数,即模态数

    "classes": 35,  # 模型最后输出的通道数,即类别总数

    "with_pmfs_block": False,  # 加不加PMFS模块

    "index_to_class_dict":  # 类别索引映射到类别名称的字典
        {
            0: "background",
            1: "gum",
            2: "implant",
            3: "ul1",
            4: "ul2",
            5: "ul3",
            6: "ul4",
            7: "ul5",
            8: "ul6",
            9: "ul7",
            10: "ul8",
            11: "ur1",
            12: "ur2",
            13: "ur3",
            14: "ur4",
            15: "ur5",
            16: "ur6",
            17: "ur7",
            18: "ur8",
            19: "bl1",
            20: "bl2",
            21: "bl3",
            22: "bl4",
            23: "bl5",
            24: "bl6",
            25: "bl7",
            26: "bl8",
            27: "br1",
            28: "br2",
            29: "br3",
            30: "br4",
            31: "br5",
            32: "br6",
            33: "br7",
            34: "br8"
        },

    "resume": None,  # 是否重启之前某个训练节点，继续训练;如果需要则指定.state文件路径

    "pretrain": None,  # 是否需要加载预训练权重，如果需要则指定预训练权重文件路径

    # ——————————————————————————————————————————————    优化器     ——————————————————————————————————————————————————————

    "optimizer_name": "AdamW",  # 优化器名称，可选["SGD", "Adagrad", "RMSprop", "Adam", "AdamW", "Adamax", "Adadelta"]

    "learning_rate": 0.05,  # 学习率

    "weight_decay": 0.00005,  # 权重衰减系数,即更新网络参数时的L2正则化项的系数

    "momentum": 0.8,  # 动量大小

    # ———————————————————————————————————————————    学习率调度器     —————————————————————————————————————————————————————

    "lr_scheduler_name": "ReduceLROnPlateau",  # 学习率调度器名称，可选["ExponentialLR", "StepLR", "MultiStepLR",
    # "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "OneCycleLR", "ReduceLROnPlateau"]

    "gamma": 0.1,  # 学习率衰减系数

    "step_size": 9,  # StepLR的学习率衰减步长

    "milestones": [1, 3, 5, 7, 8, 9],  # MultiStepLR的学习率衰减节点列表

    "T_max": 2,  # CosineAnnealingLR的半周期

    "T_0": 2,  # CosineAnnealingWarmRestarts的周期

    "T_mult": 2,  # CosineAnnealingWarmRestarts的周期放大倍数

    "mode": "max",  # ReduceLROnPlateau的衡量指标变化方向

    "patience": 1,  # ReduceLROnPlateau的衡量指标可以停止优化的最长epoch

    "factor": 0.5,  # ReduceLROnPlateau的衰减系数

    # ————————————————————————————————————————————    损失函数     ———————————————————————————————————————————————————————

    "metric_names": ["HD", "ASSD", "IoU", "SO", "DSC"],  # 采用除了dsc之外的评价指标，可选["HD", "ASSD", "IoU", "SO", "DSC"]

    "loss_function_name": "DiceLoss",  # 损失函数名称，可选["DiceLoss","CrossEntropyLoss","WeightedCrossEntropyLoss",
    # "MSELoss","SmoothL1Loss","L1Loss","WeightedSmoothL1Loss","BCEDiceLoss","BCEWithLogitsLoss"]

    "class_weight": [2.065737036158631e-05, 0.0002288518250596617, 0.10644896264800474, 0.024517091342999286, 0.031551274637690946, 0.02142641990334763, 0.023504830235874675, 0.024805246670275265,
                     0.011253835681459618, 0.012061084791323883, 0.07426874930669891, 0.025837421856808762, 0.030593882798001362, 0.02485595457346088, 0.02466779443285936, 0.025299810905129824,
                     0.011971753925834416, 0.012728768474636025, 0.16020725882894402, 0.05647514115391198, 0.028563301116552024, 0.01808694103472792, 0.021247037482269582, 0.021758918566911867,
                     0.018020924460142022, 0.015630351326653497, 0.0, 0.05555089745279102, 0.027478460253826026, 0.01756969204704165, 0.02183707476398382, 0.019346772849462818, 0.018484194657617598,
                     0.013700642625337286, 0.0],  # 各类别计算损失值的加权权重

    "sigmoid_normalization": False,  # 对网络输出的各通道进行归一化的方式,True是对各元素进行sigmoid,False是对所有通道进行softmax

    "dice_loss_mode": "extension",  # Dice Loss的计算方式，"standard":标准计算方式；"extension":扩展计算方式

    "dice_mode": "standard",  # DSC的计算方式，"standard":标准计算方式；"extension":扩展计算方式

    # —————————————————————————————————————————————   训练相关参数   ——————————————————————————————————————————————————————

    "fold_k": 5,  # k折交叉验证的折数

    "start_fold": 0,  # 从第几折开始训练
    "current_fold": 0,  # 当前是第几折

    "metric_results_per_fold": {"HD": [], "ASSD": [], "IoU": [], "SO": [], "DSC": []},  # 存储所有评价指标在每一折的结果

    "run_dir": r"./runs",  # 运行时产生的各类文件的存储根目录

    "start_epoch": 0,  # 训练时的起始epoch
    "end_epoch": 20,  # 训练时的结束epoch

    "best_dsc": -1.0,  # 保存检查点的初始条件

    "update_weight_freq": 32,  # 每多少个step更新一次网络权重，用于梯度累加

    "terminal_show_freq": 64,  # 终端打印统计信息的频率,以step为单位

    # ————————————————————————————————————————————   测试相关参数   ———————————————————————————————————————————————————————

    "crop_stride": [32, 32, 32]
}


def load_kfold_state():
    # 加载训练状态字典
    resume_state_dict = torch.load(params["resume"], map_location=lambda storage, loc: storage.cuda(params["device"]))
    # 加载交叉验证相关参数
    params["start_fold"] = resume_state_dict["fold"]
    params["current_fold"] = resume_state_dict["fold"]
    params["metric_results_per_fold"] = resume_state_dict["metric_results_per_fold"]


def cross_validation(loss_function, metric):
    # 初始化k折交叉验证
    kf = KFold(n_splits=params["fold_k"], shuffle=True, random_state=params["seed"])
    # 获取数据集中所有原图图像和标注图像的路径
    images_path_list = sorted(glob.glob(os.path.join(params["dataset_path"], "images", "*.nrrd")))
    labels_path_list = sorted(glob.glob(os.path.join(params["dataset_path"], "labels", "*.nrrd")))
    for i, (train_index, valid_index) in enumerate(kf.split(images_path_list, labels_path_list)):
        if i < params["start_fold"]:
            continue
        params["current_fold"] = i
        print("开始训练{}-fold......".format(i))
        utils.pre_write_txt("开始训练{}-fold......".format(i), params["log_txt_path"])

        # 划分数据集
        train_images_path_list, train_labels_path_list = list(np.array(images_path_list)[train_index]), list(np.array(labels_path_list)[train_index])
        valid_images_path_list, valid_labels_path_list = list(np.array(images_path_list)[valid_index]), list(np.array(labels_path_list)[valid_index])
        print([os.path.basename(path) for path in train_images_path_list], [os.path.basename(path) for path in valid_images_path_list])

        # 初始化数据加载器
        train_loader, valid_loader = dataloaders.get_dataloader(params, train_images_path_list, train_labels_path_list, valid_images_path_list, valid_labels_path_list)

        # 初始化模型、优化器和学习率调整器
        model, optimizer, lr_scheduler = models.get_model_optimizer_lr_scheduler(params)

        # 初始化训练器
        trainer = trainers.Trainer(params, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)

        # 如果需要继续训练或者加载预训练权重
        if (params["resume"] is not None) and (params["pretrain"] is not None) and (i == params["start_fold"]):
            trainer.load()

        # 开始训练
        trainer.training()

        # 释放不用的内存
        gc.collect()


def calculate_metrics():
    result_str = "\n\n"
    for metric_name, values in params["metric_results_per_fold"].items():
        result_str += metric_name + ":"
        for value in values:
            result_str += "  " + str(value)
        result_str += "\n"
    utils.pre_write_txt(result_str, params["log_txt_path"])

    print_info = "\n\n"
    # 评价指标作为列名
    print_info += " " * 12
    for metric_name in params["metric_names"]:
        print_info += "{:^12}".format(metric_name)
    print_info += '\n'
    # 添加各评价指标的均值
    print_info += "{:<12}".format("mean:")
    for metric_name, values in params["metric_results_per_fold"].items():
        print_info += "{:^12.6f}".format(np.mean(np.array(values)))
    print_info += '\n'
    # 添加各评价指标的标准差
    print_info += "{:<12}".format("std:")
    for metric_name, values in params["metric_results_per_fold"].items():
        print_info += "{:^12.6f}".format(np.std(np.array(values)))
    print(print_info)
    utils.pre_write_txt(print_info, params["log_txt_path"])


if __name__ == '__main__':

    # 设置可用GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    # 随机种子、卷积算法优化
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])

    # 获取GPU设备
    if params["cuda"]:
        params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        params["device"] = torch.device("cpu")
    print(params["device"])
    print("完成初始化配置")

    # 初始化损失函数
    loss_function = losses.get_loss_function(params)
    print("完成初始化损失函数")

    # 初始化各评价指标
    metric = metrics.get_metric(params)
    print("完成初始化评价指标")

    # 是否加载kfold状态
    if (params["resume"] is not None) and (params["pretrain"] is not None):
        load_kfold_state()
        print("完成kfold状态加载")

    # 获取训练中间文件存储路径
    if params["resume"] is None:
        params["execute_dir"] = os.path.join(params["run_dir"], utils.datestr() + "_" + params["model_name"] + "_" + params["dataset_name"])
    else:
        params["execute_dir"] = os.path.dirname(os.path.dirname(params["resume"]))
    params["checkpoint_dir"] = os.path.join(params["execute_dir"], "checkpoints")
    params["log_txt_path"] = os.path.join(params["execute_dir"], "log.txt")
    if params["resume"] is None:
        utils.make_dirs(params["checkpoint_dir"])

    # 交叉验证训练
    print("开始交叉验证训练......")
    cross_validation(loss_function, metric)

    # 计算评价指标的综合结果
    calculate_metrics()
