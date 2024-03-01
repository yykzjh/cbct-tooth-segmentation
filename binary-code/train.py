# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/1 22:52
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
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

import nni

from lib import utils, dataloaders, models, losses, metrics, trainers

# 默认参数,这里的参数在后面添加到模型中，以params['dropout_rate']等替换原来的参数
params = {
    # ——————————————————————————————————————————————     启动初始化    ———————————————————————————————————————————————————

    "CUDA_VISIBLE_DEVICES": "0",  # 选择可用的GPU编号

    "seed": 20240126,  # 随机种子

    "cuda": True,  # 是否使用GPU

    "benchmark": False,  # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。用于网络输入维度和类型不会变的情况。

    "deterministic": True,  # 固定cuda的随机数种子，每次返回的卷积算法将是确定的。用于复现模型结果

    # —————————————————————————————————————————————     预处理       ————————————————————————————————————————————————————

    "resample_spacing": [0.5, 0.5, 0.5],  # 重采样的体素间距。三个维度的值一般相等，可设为0.5(图像尺寸有[200,200,100]、[200,200,200]、
    # [160,160,160]),或者设为0.25(图像尺寸有[400,400,200]、[400,400,400]、[320,320,320])

    "clip_lower_bound": -1436,  # clip的下边界数值
    "clip_upper_bound": 17869,  # clip的上边界数值

    "samples_train": 2048,  # 作为实际的训练集采样的子卷数量，也就是在原训练集上随机裁剪的子图像数量

    "crop_size": (160, 160, 96),  # 随机裁剪的尺寸。1、每个维度都是32的倍数这样在下采样时不会报错;2、11G的显存最大尺寸不能超过(192,192,160);
    # 3、要依据上面设置的"resample_spacing",在每个维度随机裁剪的尺寸不能超过图像重采样后的尺寸;

    "crop_threshold": 0.5,  # 随机裁剪时需要满足的条件，不满足则重新随机裁剪的位置。条件表示的是裁剪区域中的前景占原图总前景的最小比例

    # ——————————————————————————————————————————————    数据增强    ——————————————————————————————————————————————————————

    "augmentation_probability": 0.3,  # 每张图像做数据增强的概率
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
    "normalize_mean": 0.05192686292050685,
    "normalize_std": 0.028678952497449627,

    # —————————————————————————————————————————————    数据读取     ——————————————————————————————————————————————————————

    "dataset_name": "BINARY-TOOTH-FULL",  # 数据集名称， 可选["BINARY-TOOTH-FULL", "BINARY-TOOTH-SURFACE"]

    "dataset_path": r"./datasets/NC-release-data-full",  # 数据集路径

    "create_data": False,  # 是否重新分割子卷训练集

    "batch_size": 1,  # batch_size大小

    "num_workers": 2,  # num_workers大小

    # —————————————————————————————————————————————    网络模型     ——————————————————————————————————————————————————————

    "model_name": "PMFSNet",  # 模型名称，可选["DenseVNet","UNet3D", "VNet", "AttentionUNet3D", "R2UNet", "R2AttentionUNet",
    # "HighResNet3D", "DenseVoxelNet", "MultiResUNet3D", "DenseASPPUNet", "PMFSNet", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet"]

    "in_channels": 1,  # 模型最开始输入的通道数,即模态数

    "classes": 2,  # 模型最后输出的通道数,即类别总数

    "scaling_version": "TINY",  # PMFSNet模型的缩放版本，可选["TINY", "SMALL", "BASIC"]

    "with_pmfs_block": False,  # 加不加PMFS模块

    "two_stage": False,  # 是否采用两阶段架构

    "surface_pretrain": None,  # 表面轮廓分割模型预训练权重

    "centroid_pretrain": None,  # 几何中心分割模型预训练权重

    "index_to_class_dict":  # 类别索引映射到类别名称的字典
        {
            0: "background",
            1: "foreground"
        },

    "resume": None,  # 是否重启之前某个训练节点，继续训练;如果需要则指定.state文件路径

    "pretrain": None,  # 是否需要加载预训练权重，如果需要则指定预训练权重文件路径

    # ——————————————————————————————————————————————    优化器     ——————————————————————————————————————————————————————

    "optimizer_name": "Adam",  # 优化器名称，可选["SGD", "Adagrad", "RMSprop", "Adam", "AdamW", "Adamax", "Adadelta"]

    "learning_rate": 0.0005,  # 学习率

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

    "class_weight": [0.006082026617935588, 0.9939179733820644],  # 各类别计算损失值的加权权重

    "sigmoid_normalization": False,  # 对网络输出的各通道进行归一化的方式,True是对各元素进行sigmoid,False是对所有通道进行softmax

    "dice_loss_mode": "extension",  # Dice Loss的计算方式，"standard":标准计算方式；"extension":扩展计算方式

    "dice_mode": "standard",  # DSC的计算方式，"standard":标准计算方式；"extension":扩展计算方式

    # —————————————————————————————————————————————   训练相关参数   ——————————————————————————————————————————————————————

    "optimize_params": False,  # 程序是否处于优化参数的模型，不需要保存训练的权重和中间结果

    "run_dir": r"./runs",  # 运行时产生的各类文件的存储根目录

    "start_epoch": 0,  # 训练时的起始epoch
    "end_epoch": 20,  # 训练时的结束epoch

    "best_dsc": 0.0,  # 保存检查点的初始条件

    "update_weight_freq": 32,  # 每多少个step更新一次网络权重，用于梯度累加

    "terminal_show_freq": 256,  # 终端打印统计信息的频率,以step为单位

    "save_epoch_freq": 30,  # 每多少个epoch保存一次训练状态和模型参数

    # ————————————————————————————————————————————   测试相关参数   ———————————————————————————————————————————————————————

    "crop_stride": [32, 32, 32]
}

if __name__ == '__main__':

    if params["optimize_params"]:
        # 获得下一组搜索空间中的参数
        tuner_params = nni.get_next_parameter()
        # 更新参数
        params.update(tuner_params)

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

    # 初始化数据加载器
    train_loader, valid_loader = dataloaders.get_dataloader(params)
    print("完成初始化数据加载器")

    # 初始化模型、优化器和学习率调整器
    model, optimizer, lr_scheduler = models.get_model_optimizer_lr_scheduler(params)
    print("完成初始化模型:{}、优化器:{}和学习率调整器:{}".format(params["model_name"], params["optimizer_name"], params["lr_scheduler_name"]))

    # 初始化损失函数
    loss_function = losses.get_loss_function(params)
    print("完成初始化损失函数")

    # 初始化各评价指标
    metric = metrics.get_metric(params)
    print("完成初始化评价指标")

    # 创建训练执行目录和文件
    if not params["optimize_params"]:
        if params["resume"] is None:
            stage_str = ("_Two-Stage" if params["two_stage"] else "_Single-Stage")
            pmfs_str = ("_With-PMFS" if params["with_pmfs_block"] else "_No-PMFS")
            params["execute_dir"] = os.path.join(params["run_dir"],
                                                 utils.datestr() +
                                                 stage_str +
                                                 "_" + params["model_name"] +
                                                 pmfs_str +
                                                 "_" + params["dataset_name"])
        else:
            params["execute_dir"] = os.path.dirname(os.path.dirname(params["resume"]))
        params["checkpoint_dir"] = os.path.join(params["execute_dir"], "checkpoints")
        params["tensorboard_dir"] = os.path.join(params["execute_dir"], "board")
        params["log_txt_path"] = os.path.join(params["execute_dir"], "log.txt")
        if params["resume"] is None:
            utils.make_dirs(params["checkpoint_dir"])
            utils.make_dirs(params["tensorboard_dir"])

    # 初始化训练器
    trainer = trainers.Trainer(params, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)

    # 如果需要继续训练或者加载预训练权重
    if (params["resume"] is not None) or (params["pretrain"] is not None):
        trainer.load()

    # 开始训练
    print("开始训练......")
    trainer.training()
