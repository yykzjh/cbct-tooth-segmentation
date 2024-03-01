# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/1 22:50
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch
import torch.optim as optim

import lib.utils as utils

from .DenseVNet import DenseVNet
from .UNet3D import UNet3D
from .VNet import VNet
from .AttentionUNet3D import AttentionUNet3D
from .R2UNet import R2U_Net
from .R2AttentionUNet import R2AttentionU_Net
from .HighResNet3D import HighResNet3D
from .DenseVoxelNet import DenseVoxelNet
from .MultiResUNet3D import MultiResUNet3D
from .DenseASPPUNet import DenseASPPUNet
from .UNETR import UNETR
from .SwinUNETR import SwinUNETR
from .TransBTS import BTS
from lib.models.nnFormer.nnFormer_seg import nnFormer
from lib.models.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import AttentionUnet

from .PMFSNet import PMFSNet

from .TwoStageNet import TwoStageNet


def get_model_optimizer_lr_scheduler(opt):
    # 初始化网络模型
    if opt["model_name"] == "DenseVNet":
        model = DenseVNet(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), classes=opt["classes"], with_pmfs_block=opt["with_pmfs_block"])

    elif opt["model_name"] == "UNet3D":
        model = UNet3D(opt["in_channels"] + (2 if opt["two_stage"] else 0), opt["classes"], final_sigmoid=False, with_pmfs_block=opt["with_pmfs_block"])

    elif opt["model_name"] == "VNet":
        model = VNet(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), classes=opt["classes"], with_pmfs_block=opt["with_pmfs_block"])

    elif opt["model_name"] == "AttentionUNet3D":
        model = AttentionUNet3D(spatial_dims=3, in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), out_channels=opt["classes"], with_pmfs_block=opt["with_pmfs_block"])
        # model = AttentionUnet(spatial_dims=3, in_channels=opt["in_channels"]+(2 if opt["two_stage"] else 0), out_channels=opt["classes"], channels=(64, 128, 256, 512, 1024), strides=(2, 2, 2, 2))

    elif opt["model_name"] == "R2UNet":
        model = R2U_Net(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), out_channels=opt["classes"])

    elif opt["model_name"] == "R2AttentionUNet":
        model = R2AttentionU_Net(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), out_channels=opt["classes"])

    elif opt["model_name"] == "HighResNet3D":
        model = HighResNet3D(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), classes=opt["classes"])

    elif opt["model_name"] == "DenseVoxelNet":
        model = DenseVoxelNet(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), classes=opt["classes"], with_pmfs_block=opt["with_pmfs_block"])

    elif opt["model_name"] == "MultiResUNet3D":
        model = MultiResUNet3D(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), classes=opt["classes"], with_pmfs_block=opt["with_pmfs_block"])

    elif opt["model_name"] == "DenseASPPUNet":
        model = DenseASPPUNet(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), classes=opt["classes"])

    elif opt["model_name"] == "UNETR":
        model = UNETR(
            in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0),
            out_channels=opt["classes"],
            img_size=(160, 160, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
            with_pmfs_block=opt["with_pmfs_block"]
        )

    elif opt["model_name"] == "SwinUNETR":
        model = SwinUNETR(
            img_size=(160, 160, 96),
            in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0),
            out_channels=opt["classes"],
            feature_size=48,
            use_checkpoint=False,
            with_pmfs_block=opt["with_pmfs_block"]
        )

    elif opt["model_name"] == "TransBTS":
        model = BTS(img_dim=(160, 160, 96), patch_dim=8, num_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), num_classes=opt["classes"],
                    embedding_dim=512,
                    num_heads=8,
                    num_layers=4,
                    hidden_dim=4096,
                    dropout_rate=0.1,
                    attn_dropout_rate=0.1,
                    conv_patch_representation=True,
                    positional_encoding_type="learned",
                    with_pmfs_block=opt["with_pmfs_block"]
                    )

    elif opt["model_name"] == "nnFormer":
        model = nnFormer(crop_size=(160, 160, 96), input_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), num_classes=opt["classes"], with_pmfs_block=opt["with_pmfs_block"])

    elif opt["model_name"] == "3DUXNet":
        model = UXNET(
            in_chans=opt["in_channels"] + (2 if opt["two_stage"] else 0),
            out_chans=opt["classes"],
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
            with_pmfs_block=opt["with_pmfs_block"]
        )

    elif opt["model_name"] == "PMFSNet":
        model = PMFSNet(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), out_channels=opt["classes"], dim="3d", scaling_version=opt["scaling_version"], with_pmfs_block=opt["with_pmfs_block"])

    else:
        raise RuntimeError(f"{opt['model_name']}是不支持的网络模型！")

    # 把模型放到GPU上
    model = model.to(opt["device"])

    # 随机初始化模型参数
    utils.init_weights(model, init_type="kaiming")

    # 初始化优化器
    if opt["optimizer_name"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=opt["learning_rate"], momentum=opt["momentum"],
                              weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"],
                                  momentum=opt["momentum"])

    elif opt["optimizer_name"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    else:
        raise RuntimeError(
            f"{opt['optimizer_name']}是不支持的优化器！")

    # 初始化学习率调度器
    if opt["lr_scheduler_name"] == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "MultiStepLR":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt["milestones"], gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt["T_max"])

    elif opt["lr_scheduler_name"] == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt["T_0"],
                                                                      T_mult=opt["T_mult"])

    elif opt["lr_scheduler_name"] == "OneCycleLR":
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt["learning_rate"],
                                                     steps_per_epoch=opt["steps_per_epoch"], epochs=opt["end_epoch"], cycle_momentum=False)

    elif opt["lr_scheduler_name"] == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=opt["mode"], factor=opt["factor"],
                                                            patience=opt["patience"])
    else:
        raise RuntimeError(
            f"{opt['lr_scheduler_name']}是不支持的学习率调度器！")

    # 判断是否采用两阶段架构
    if opt["two_stage"]:
        model = TwoStageNet(opt, model, in_channels=opt["in_channels"])

    return model, optimizer, lr_scheduler


def get_model(opt):
    # 初始化网络模型
    if opt["model_name"] == "DenseVNet":
        model = DenseVNet(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), classes=opt["classes"], with_pmfs_block=opt["with_pmfs_block"])

    elif opt["model_name"] == "UNet3D":
        model = UNet3D(opt["in_channels"] + (2 if opt["two_stage"] else 0), opt["classes"], final_sigmoid=False, with_pmfs_block=opt["with_pmfs_block"])

    elif opt["model_name"] == "VNet":
        model = VNet(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), classes=opt["classes"], with_pmfs_block=opt["with_pmfs_block"])

    elif opt["model_name"] == "AttentionUNet3D":
        model = AttentionUNet3D(spatial_dims=3, in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), out_channels=opt["classes"], with_pmfs_block=opt["with_pmfs_block"])
        # model = AttentionUnet(spatial_dims=3, in_channels=opt["in_channels"]+(2 if opt["two_stage"] else 0), out_channels=opt["classes"], channels=(64, 128, 256, 512, 1024), strides=(2, 2, 2, 2))

    elif opt["model_name"] == "R2UNet":
        model = R2U_Net(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), out_channels=opt["classes"])

    elif opt["model_name"] == "R2AttentionUNet":
        model = R2AttentionU_Net(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), out_channels=opt["classes"])

    elif opt["model_name"] == "HighResNet3D":
        model = HighResNet3D(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), classes=opt["classes"])

    elif opt["model_name"] == "DenseVoxelNet":
        model = DenseVoxelNet(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), classes=opt["classes"], with_pmfs_block=opt["with_pmfs_block"])

    elif opt["model_name"] == "MultiResUNet3D":
        model = MultiResUNet3D(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), classes=opt["classes"], with_pmfs_block=opt["with_pmfs_block"])

    elif opt["model_name"] == "DenseASPPUNet":
        model = DenseASPPUNet(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), classes=opt["classes"])

    elif opt["model_name"] == "UNETR":
        model = UNETR(
            in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0),
            out_channels=opt["classes"],
            img_size=(160, 160, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
            with_pmfs_block=opt["with_pmfs_block"]
        )

    elif opt["model_name"] == "SwinUNETR":
        model = SwinUNETR(
            img_size=(160, 160, 96),
            in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0),
            out_channels=opt["classes"],
            feature_size=48,
            use_checkpoint=False,
            with_pmfs_block=opt["with_pmfs_block"]
        )

    elif opt["model_name"] == "TransBTS":
        model = BTS(img_dim=(160, 160, 96), patch_dim=8, num_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), num_classes=opt["classes"],
                    embedding_dim=512,
                    num_heads=8,
                    num_layers=4,
                    hidden_dim=4096,
                    dropout_rate=0.1,
                    attn_dropout_rate=0.1,
                    conv_patch_representation=True,
                    positional_encoding_type="learned",
                    with_pmfs_block=opt["with_pmfs_block"]
                    )

    elif opt["model_name"] == "nnFormer":
        model = nnFormer(crop_size=(160, 160, 96), input_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), num_classes=opt["classes"], with_pmfs_block=opt["with_pmfs_block"])

    elif opt["model_name"] == "3DUXNet":
        model = UXNET(
            in_chans=opt["in_channels"] + (2 if opt["two_stage"] else 0),
            out_chans=opt["classes"],
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
            with_pmfs_block=opt["with_pmfs_block"]
        )

    elif opt["model_name"] == "PMFSNet":
        model = PMFSNet(in_channels=opt["in_channels"] + (2 if opt["two_stage"] else 0), out_channels=opt["classes"], dim="3d", scaling_version=opt["scaling_version"], with_pmfs_block=opt["with_pmfs_block"])

    else:
        raise RuntimeError(f"{opt['model_name']}是不支持的网络模型！")

    # 把模型放到GPU上
    model = model.to(opt["device"])

    # 判断是否采用两阶段架构
    if opt["two_stage"]:
        model = TwoStageNet(opt, model, in_channels=opt["in_channels"])

    # 随机初始化模型参数
    utils.init_weights(model, init_type="kaiming")

    return model
