# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/7 23:41
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import glob
import math
import tqdm
import time
import shutil
import random
import cv2
import numpy as np
import pandas as pd
import trimesh
import json
import torch
from torchinfo import summary
from thop import profile
from ptflops import get_model_complexity_info
from torchstat import stat
import nibabel as nib
from tqdm import tqdm
from functools import reduce
from nibabel.viewers import OrthoSlicer3D
import SimpleITK as sitk
from proplot import rc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import proplot as pplt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

import lib.utils as utils
import lib.models as models

index_to_class_dict = {
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
}


def load_nii_file(file_path):
    """
    底层读取.nii.gz文件
    Args:
        file_path: 文件路径
        type: image或者label

    Returns: uint16格式的三维numpy数组, spacing三个元素的元组

    """
    image_np, spacing = utils.load_nii_file(file_path)

    print(image_np.min(), image_np.max())
    print(spacing)


def load_obj_file(file_path):
    """
    底层读取.obj文件
    Args:
        file_path: 文件路径

    Returns:
    """
    obj = trimesh.load(file_path, process=False)
    v = obj.vertices
    f = obj.faces
    v1 = np.array(v)
    f1 = np.array(f)
    print(v1.shape)
    print(f1.shape)

    json_file_path = os.path.splitext(file_path)[0] + ".json"
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_file = json.load(f)
    labels = json_file["labels"]
    print(len(labels))


def split_dataset(dataset_dir, train_ratio=0.8, seed=123):
    # 设置随机种子
    np.random.seed(seed)
    # 创建训练集和验证集文件夹
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "valid")
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    os.makedirs(os.path.join(train_dir, "images"))
    os.makedirs(os.path.join(train_dir, "labels"))
    os.makedirs(os.path.join(val_dir, "images"))
    os.makedirs(os.path.join(val_dir, "labels"))
    # 读取所有的
    images_path_list = glob.glob(os.path.join(dataset_dir, "image", "*.nii.gz"))
    labels_path_list = glob.glob(os.path.join(dataset_dir, "label", "*.nii.gz"))
    # 随机抽样训练集
    trainset_list = random.sample(images_path_list, int(len(images_path_list) * 0.8))
    valset_list = [path for path in images_path_list if path not in trainset_list]
    # 复制训练集
    for path in tqdm(trainset_list):
        file_name = os.path.basename(path)
        dest_image_path = os.path.join(train_dir, "images", file_name)
        # 复制原图
        shutil.copyfile(path, dest_image_path)
        src_label_path = path.replace("image", "label")
        dest_label_path = os.path.join(train_dir, "labels", file_name)
        # 复制标签图
        shutil.copyfile(src_label_path, dest_label_path)
    # 复制验证集
    for path in tqdm(valset_list):
        file_name = os.path.basename(path)
        dest_image_path = os.path.join(val_dir, "images", file_name)
        # 复制原图
        shutil.copyfile(path, dest_image_path)
        src_label_path = path.replace("image", "label")
        dest_label_path = os.path.join(val_dir, "labels", file_name)
        # 复制标签图
        shutil.copyfile(src_label_path, dest_label_path)


def analyse_image_label_consistency(dataset_dir, resample_spacing=(0.5, 0.5, 0.5)):
    # 定义合格数据集存储根目录
    new_dataset_dir = os.path.join(os.path.dirname(dataset_dir), os.path.basename(dataset_dir) + "-checked")
    new_dataset_image_dir = os.path.join(new_dataset_dir, "image")
    new_dataset_label_dir = os.path.join(new_dataset_dir, "label")
    # 创建新数据集目录
    if os.path.exists(new_dataset_dir):
        shutil.rmtree(new_dataset_dir)
    os.makedirs(new_dataset_dir)
    os.makedirs(new_dataset_image_dir)
    os.makedirs(new_dataset_label_dir)

    # 读取数据集路径列表
    images_list = sorted(glob.glob(os.path.join(dataset_dir, "image", "*.nii.gz")))
    labels_list = sorted(glob.glob(os.path.join(dataset_dir, "label", "*.nii.gz")))
    # 条件一：首先检测image和label数据集大小是否一致
    assert len(images_list) == len(labels_list), "image和label数据集大小不一致"

    # 遍历原images
    for image_path in tqdm(images_list):
        # 首先当前image要有同名的label文件
        virtual_label_path = image_path.replace("image", "label")  # 创建虚拟同名label的路径
        # 条件二：如果没有该label，跳过当前image
        if virtual_label_path not in labels_list:
            continue

        # 如果当前image能找到同名的label，则都先读取
        image_np = utils.load_image_or_label(image_path, resample_spacing, type="image")
        label_np = utils.load_image_or_label(virtual_label_path, resample_spacing, type="label")
        # 条件三：判断重采样后的image和label图像尺寸是否一致，不一致则跳过当前image
        if image_np.shape != label_np.shape:
            continue

        # 最后分析label标注的前景是否有较大概率是image中的牙齿来判断标注文件的正确性
        # 首先获取image中的最小值和最大值
        min_val = image_np.min()
        max_val = image_np.max()
        # 获取原image的拷贝
        image_copy = image_np.copy()
        # 将image_copy的值归一化到0~255
        image_copy = (image_copy - min_val) / (max_val - min_val) * 255
        # 转换为整型
        image_copy = image_copy.astype(np.int32)
        # 获取骨骼和软组织的阈值
        T0, T1 = utils.get_multi_threshold(image_copy, m=2, max_length=255)
        # 展示分割点
        # plt.hist(image_copy.flatten(), bins=256, color="g", histtype="bar", rwidth=1, alpha=0.6)
        # plt.axvline(T1, color="r")
        # plt.show()
        # 获得阈值在原图中的值
        ori_T = T1 / 255 * (max_val - min_val) + min_val
        # 通过label可知所有前景元素，计算前景部分的值大于ori_T的比例
        ratio = np.sum(image_np[label_np > 0] > ori_T) / np.sum(label_np > 0)
        # 条件四：如果比例小于0.95，则认为当前label图像标注有问题，跳过当前image
        if ratio < 0.90:
            continue

        # 如果前面的四个条件都满足，则将当前image和label存储到新数据集位置
        file_name = os.path.basename(image_path)  # 获取文件名
        # 转移存储
        shutil.copyfile(image_path, os.path.join(new_dataset_image_dir, file_name))
        shutil.copyfile(virtual_label_path, os.path.join(new_dataset_label_dir, file_name))


def analyse_dataset(dataset_dir, resample_spacing=(0.5, 0.5, 0.5), clip_lower_bound_ratio=0.01, clip_upper_bound_ratio=0.995, classes=2):
    # 加载数据集中所有原图像
    images_list = sorted(glob.glob(os.path.join(dataset_dir, "images", "*.nrrd")))
    # 加载数据集中所有标注图像
    labels_list = sorted(glob.glob(os.path.join(dataset_dir, "labels", "*.nrrd")))
    assert len(images_list) == len(labels_list), "原图像数量和标注图像数量不一致"

    print("开始统计体素总量、下界百分位到最小值的长度、上界百分位到最大值的长度->")
    # 先统计数据集的总体素量
    total_voxels = 0
    # 遍历所有图像
    for i in tqdm(range(len(images_list))):
        # 判断一下原图像文件名和标注图像文件名一致
        assert os.path.basename(images_list[i]) == os.path.basename(labels_list[i]), "原图像文件名和标注图像文件名不一致"
        # 获取当前图像的原图像和标注图像
        image_np = utils.load_image_or_label(images_list[i], resample_spacing, type="image")
        # 累计体素
        total_voxels += reduce(lambda acc, cur: acc * cur, image_np.shape)
    # 计算上下界分位点需要存储的数据量
    lower_length = int(total_voxels * clip_lower_bound_ratio + 1)
    upper_length = int(total_voxels - total_voxels * clip_upper_bound_ratio + 1)
    print("体素总量:{}, 下界百分位到最小值的长度:{}, 上界百分位到最大值的长度:{}".format(total_voxels, lower_length, upper_length))

    print("开始计算所有元素和前景元素的最小值和最大值、clip的上下界->")
    # 初始化数据结构
    lower_all_values = np.array([])
    upper_all_values = np.array([])
    all_values_min = 1e9
    all_values_max = -1e9
    foreground_values_min = 1e9
    foreground_values_max = -1e9
    # 遍历所有图像
    for i in tqdm(range(len(images_list))):
        # 判断一下原图像文件名和标注图像文件名一致
        assert os.path.basename(images_list[i]) == os.path.basename(labels_list[i]), "原图像文件名和标注图像文件名不一致"
        # 获取当前图像的原图像和标注图像
        image_np = utils.load_image_or_label(images_list[i], resample_spacing, type="image")
        label_np = utils.load_image_or_label(labels_list[i], resample_spacing, type="label", index_to_class_dict=index_to_class_dict)
        # 将当前图像的体素添加到上下界的存储数组中并排序
        lower_all_values = np.sort(np.concatenate([lower_all_values, image_np.flatten()], axis=0))
        upper_all_values = np.concatenate([upper_all_values, image_np.flatten()], axis=0)
        upper_all_values = upper_all_values[np.argsort(-upper_all_values)]
        # 判断需不需要裁剪所需的部分
        if len(lower_all_values) > lower_length:
            lower_all_values = lower_all_values[:lower_length]
        if len(upper_all_values) > upper_length:
            upper_all_values = upper_all_values[:upper_length]
        # 维护全部体素和前景体素的最小值和最大值
        all_values_min = min(all_values_min, image_np.min())
        all_values_max = max(all_values_max, image_np.max())
        foreground_values_min = min(foreground_values_min, image_np[label_np != 0].min())
        foreground_values_max = max(foreground_values_max, image_np[label_np != 0].max())

    # 计算指定的上下界分位点的灰度值
    clip_lower_bound = lower_all_values[lower_length - 1]
    clip_upper_bound = upper_all_values[upper_length - 1]

    # 输出所有元素和前景元素的最小值和最大值
    print("all_values_min:{}, fore_values_min:{}, fore_values_max:{}, all_values_max:{}"
          .format(all_values_min, foreground_values_min, foreground_values_max, all_values_max))
    # 输出clip的上下界
    print("clip_lower_bound:{}, clip_upper_bound:{}".format(clip_lower_bound, clip_upper_bound))

    print("开始计算均值和方差->")
    # 初始化均值的加和
    mean_sum = 0
    # 遍历所有图像
    for i in tqdm(range(len(images_list))):
        # 判断一下原图像文件名和标注图像文件名一致
        assert os.path.basename(images_list[i]) == os.path.basename(labels_list[i]), "原图像文件名和标注图像文件名不一致"
        # 获取当前图像的原图像
        image_np = utils.load_image_or_label(images_list[i], resample_spacing, type="image")
        # 对所有灰度值进行clip
        image_np[image_np < clip_lower_bound] = clip_lower_bound
        image_np[image_np > clip_upper_bound] = clip_upper_bound
        # 先将当前图像进行归一化
        image_np = (image_np - clip_lower_bound) / (clip_upper_bound - clip_lower_bound)
        # 累加
        mean_sum += np.sum(image_np)
    # 计算均值
    mean = mean_sum / total_voxels

    # 初始化标准差的加和
    std_sum = 0
    # 遍历所有图像
    for i in tqdm(range(len(images_list))):
        # 判断一下原图像文件名和标注图像文件名一致
        assert os.path.basename(images_list[i]) == os.path.basename(labels_list[i]), "原图像文件名和标注图像文件名不一致"
        # 获取当前图像的原图像
        image_np = utils.load_image_or_label(images_list[i], resample_spacing, type="image")
        # 对所有灰度值进行clip
        image_np[image_np < clip_lower_bound] = clip_lower_bound
        image_np[image_np > clip_upper_bound] = clip_upper_bound
        # 先将当前图像进行归一化
        image_np = (image_np - clip_lower_bound) / (clip_upper_bound - clip_lower_bound)
        # 计算当前图像每个灰度值减去均值的平方
        image_np = (image_np - mean) ** 2
        # 累加
        std_sum += np.sum(image_np)
    # 计算标准差
    std = np.sqrt(std_sum / total_voxels)

    print("均值为:{}, 标准差为:{}".format(mean, std))

    print("开始计算每个类别的权重数组->")
    # 初始化统计数组
    statistics_np = np.zeros((classes,))
    # 遍历所有图像
    for i in tqdm(range(len(images_list))):
        # 判断一下原图像文件名和标注图像文件名一致
        assert os.path.basename(images_list[i]) == os.path.basename(labels_list[i]), "原图像文件名和标注图像文件名不一致"
        # 获取当前图像的标注图像
        label_np = utils.load_image_or_label(labels_list[i], resample_spacing, type="label", index_to_class_dict=index_to_class_dict)
        # 统计在当前标注图像中出现的类别索引以及各类别索引出现的次数
        class_indexes, indexes_cnt = np.unique(label_np, return_counts=True)
        # 遍历更新到统计数组中
        for j, class_index in enumerate(class_indexes):
            # 获取当前类别索引的次数
            index_cnt = indexes_cnt[j]
            # 累加当前类别索引的次数
            statistics_np[class_index] += index_cnt

    # 初始化权重向量
    weights = np.zeros((classes,))
    # 依次计算每个类别的权重
    for i, cnt in enumerate(statistics_np):
        if cnt != 0:
            weights[i] = 1 / cnt
    # 归一化权重数组
    weights = weights / weights.sum()
    print("各类别的权重数组为：", end='[')
    weights_str = ", ".join([str(weight) for weight in weights])
    print(weights_str + "]")


def count_parameters(model):
    """计算PyTorch模型的参数数量"""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_all_models_parameters(model_names_list):
    # 先构造参数字典
    opt = {
        "in_channels": 1,
        "classes": 35,
        "device": "cuda:0",
        "scaling_version": "TINY",
        "with_pmfs_block": False,
        "two_stage": False,
        "surface_pretrain": None,
        "centroid_pretrain": None
    }
    # 遍历统计各个模型参数量
    for model_name in model_names_list:
        # 获取当前模型
        opt["model_name"] = model_name
        model = models.get_model(opt)

        print("***************************************** model name: {} *****************************************".format(model_name))

        # print("params: {:.6f}M".format(count_parameters(model)))
        #
        # input = torch.randn(1, 1, 160, 160, 96).to(opt["device"])
        # flops, params = profile(model, (input,))
        # print("flops: {:.6f}G, params: {:.6f}M".format(flops / 1e9, params / 1e6))

        flops, params = get_model_complexity_info(model, (1, 160, 160, 96), as_strings=False, print_per_layer_stat=False)
        print("flops: {:.6f} G, params: {:.6f} M".format(flops / 1e9, params / 1e6))

        x = torch.randn((1, opt["in_channels"], 160, 160, 96)).to(opt["device"])
        tsum = 0
        with torch.no_grad():
            y = model(x)
            for i in range(10):
                t1 = time.time()
                y = model(x)
                t2 = time.time()
                tsum += (t2 - t1) * 1000
        print("推理过程平均用时: {} ms".format(tsum / 10))


def generate_NC_release_data_snapshot(root_dir, size=224):
    src_image_dir = os.path.join(root_dir, "NC-release-data", "image")
    src_label_dir = os.path.join(root_dir, "NC-release-data", "label")
    dst_root_dir = os.path.join(root_dir, "snapshots")
    if os.path.exists(dst_root_dir):
        shutil.rmtree(dst_root_dir)
    os.makedirs(dst_root_dir)

    # 加载数据集中所有原图像
    images_list = sorted(glob.glob(os.path.join(src_image_dir, "*.nii.gz")))
    labels_list = sorted(glob.glob(os.path.join(src_label_dir, "*.nii.gz")))
    for i in tqdm(range(len(images_list))):
        file_name = os.path.splitext(os.path.splitext(os.path.basename(images_list[i]))[0])[0]
        image_dir = os.path.join(dst_root_dir, file_name)
        os.makedirs(image_dir)
        # image_np = utils.load_image_or_label(images_list[i], (0.5, 0.5, 0.5), type="image")
        label_np = utils.load_image_or_label(labels_list[i], (0.5, 0.5, 0.5), type="label")
        h, w, d = label_np.shape
        s = (d - 30) // 2
        for j in range(6):
            image_snapshot = label_np[:, :, s + 5 * j]
            image_snapshot = cv2.transpose(image_snapshot)
            image_snapshot = cv2.flip(image_snapshot, 0)
            # image_snapshot = np.clip(image_snapshot, -450, 1450)
            # image_snapshot = cv2.resize(image_snapshot, (size, size))
            # cv2.imwrite(os.path.join(image_dir, str(j) + ".jpg"), image_snapshot)
            plt.imshow(image_snapshot, cmap="gray")
            plt.show()


def compare_Dice():
    # 统一设置字体
    rc["font.family"] = "Times New Roman"
    # 统一设置轴刻度标签的字体大小
    rc['tick.labelsize'] = 8
    # 统一设置xy轴名称的字体大小
    rc["axes.labelsize"] = 10
    # 统一设置轴刻度标签的字体粗细
    rc["axes.labelweight"] = "light"
    # 统一设置xy轴名称的字体粗细
    rc["tick.labelweight"] = "light"

    x = np.linspace(0, 1, 1000)
    y1 = 1 - ((2 * x) / (x + 1))
    y2 = 1 - ((2 * x) / (x ** 2 + 1))

    plt.plot(x, y1, label="Standard Dice Loss")
    plt.plot(x, y2, label="Extended Dice Loss")
    plt.legend(fontsize=8)
    plt.xlabel("probability of ground truth class")
    plt.ylabel("loss")

    plt.savefig('loss.jpg', dpi=600, bbox_inches='tight')


def generate_samples_image(scale=2):
    # # 创建整个大图
    # image = np.full((960, 1295, 3), 255)
    # # 依次遍历
    # for i in range(4):
    #     for j in range(4):
    #         x = i * (224 + 8)
    #         y = j * (320 + 5)
    #         # 读取并处理图像
    #         img = cv2.imread(r"./images/challenging_samples/" + str(i) + str(j) + ".jpg")
    #         img = cv2.resize(img, (320, 224))
    #         image[x: x + 224, y: y + 320, :] = img
    # image = image[:, :, ::-1]
    # # 添加文字
    # image = Image.fromarray(np.uint8(image))
    # draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 36)
    # color = (0, 0, 0)
    #
    # position1 = (60, 917)
    # text1 = "Original Image"
    # draw.text(position1, text1, font=font, fill=color)
    #
    # position2 = (350, 917)
    # text2 = "Ground Truth (2D)"
    # draw.text(position2, text2, font=font, fill=color)
    #
    # position3 = (665, 917)
    # text3 = "Coronal MIP Image"
    # draw.text(position3, text3, font=font, fill=color)
    #
    # position4 = (990, 917)
    # text4 = "Ground Truth (3D)"
    # draw.text(position4, text4, font=font, fill=color)
    #
    # image.show()
    # w, h = image.size
    # image = image.resize((scale * w, scale * h), resample=Image.Resampling.BILINEAR)
    # print(image.size)
    # image.save(r"./images/challenging_samples/challenging_samples.jpg")

    # 创建整个大图
    image = np.full((728, 970, 3), 255)
    # 依次遍历
    for i in range(3):
        for j in range(3):
            x = i * (224 + 8)
            y = j * (320 + 5)
            # 读取并处理图像
            img = cv2.imread(r"./images/samples/" + str(i) + str(j) + ".jpg")
            img = cv2.resize(img, (320, 224))
            image[x: x + 224, y: y + 320, :] = img
    image = image[:, :, ::-1]
    # 添加文字
    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 36)
    color = (0, 0, 0)

    position1 = (60, 685)
    text1 = "Original Image"
    draw.text(position1, text1, font=font, fill=color)

    position2 = (350, 685)
    text2 = "Ground Truth (2D)"
    draw.text(position2, text2, font=font, fill=color)

    position3 = (665, 685)
    text3 = "Ground Truth (3D)"
    draw.text(position3, text3, font=font, fill=color)

    image.show()
    w, h = image.size
    image = image.resize((scale * w, scale * h), resample=Image.Resampling.BILINEAR)
    print(image.size)
    image.save(r"./images/samples/Multiple_Dataset_Samples.jpg")


def generate_segmented_sample_image(scale=1):
    # 创建整个大图
    image = np.full((976, 4340, 3), 255)
    # 依次遍历
    for i in range(4):
        for j in range(13):
            pos_x, pos_y = i * (224 + 10), j * (320 + 10) + 60
            img = cv2.imread(r"./images/NC-release-data_segment_result_samples/" + str(i) + "_{:02d}".format(j) + ".jpg")
            img = np.rot90(img, -1)
            img = cv2.resize(img, (320, 224))
            image[pos_x: pos_x + 224, pos_y: pos_y + 320, :] = img
    image = image[:, :, ::-1]

    # 添加文字的设置
    col_names = ["Image", "Ground Truth", "UNet3D", "DenseVNet", "AttentionUNet3D", "DenseVoxelNet", "MultiResUNet3D", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet", "PMFSNet"]
    row_names = ["(a)", "(b)", "(c)", "(d)"]
    col_positions = [170, 445, 820, 1125, 1410, 1755, 2075, 2470, 2765, 3110, 3455, 3780, 4110]
    row_positions = [100, 334, 568, 802]

    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 36)
    color = (0, 0, 0)

    # 遍历添加文字
    for i, text in enumerate(col_names):
        position = (col_positions[i], 931)
        draw.text(position, text, font=font, fill=color)
    for i, text in enumerate(row_names):
        position = (5, row_positions[i])
        draw.text(position, text, font=font, fill=color, stroke_width=1)

    image.show()
    w, h = image.size
    image = image.resize((scale * w, scale * h), resample=Image.Resampling.BILINEAR)
    print(image.size)
    image.save(r"./images/NC-release-data_segment_result_samples/3D_CBCT_Tooth_segmentation.jpg")


def generate_bubble_image():
    def circle_area_func(x, p=75, k=150):
        return np.where(x < p, (np.sqrt(x / p) * p) * k, x * k)

    def inverse_circle_area_func(x, p=75, k=150):
        return np.where(x < p * k, (((x / k) / p) ** 2) * p, x / k)

    rc["font.family"] = "Times New Roman"
    rc["axes.labelsize"] = 36
    rc["tick.labelsize"] = 32
    rc["suptitle.size"] = 28
    rc["title.size"] = 28

    data = pd.read_excel(r"./experience_data.xlsx", sheet_name="data01")
    model_names = data.Method
    FLOPs = data.FLOPs
    Params = data.Params
    values = data.IoU
    xtext_positions = [2250, 20, 2300, 1050, 225, 1010, 200, 600, 1810, 360, 975, 10]
    ytext_positions = [68, 80.5, 54, 73, 70, 82, 83.5, 84, 74, 78, 86.5, 86]
    legend_sizes = [1, 5, 25, 50, 100, 150]
    legend_yposition = 57.5
    legend_xpositions = [590, 820, 1030, 1260, 1480, 1710]
    p = 15
    k = 150

    # 画图
    fig, ax = plt.subplots(figsize=(18, 9), dpi=100, facecolor="w")
    pubble = ax.scatter(x=FLOPs, y=values, s=circle_area_func(Params, p=p, k=k), c=list(range(len(model_names))), cmap=plt.cm.get_cmap("Spectral"), lw=3, ec="white", vmin=0, vmax=11)
    center = ax.scatter(x=FLOPs[:-1], y=values[:-1], s=20, c="#e6e6e6")
    ours_ = ax.scatter(x=FLOPs[-1:], y=values[-1:], s=60, marker="*", c="red")

    # 添加文字
    for i in range(len(FLOPs)):
        ax.annotate(model_names[i], xy=(FLOPs[i], values[i]), xytext=(xtext_positions[i], ytext_positions[i]), fontsize=32, fontweight=(200 if i < (len(FLOPs) - 1) else 600))
    for i, legend_size in enumerate(legend_sizes):
        ax.text(legend_xpositions[i], legend_yposition, str(legend_size) + "M", fontsize=32, fontweight=200)

    # 添加图例
    kw = dict(prop="sizes", num=legend_sizes, color="#e6e6e6", fmt="{x:.0f}", linewidth=None, markeredgewidth=3, markeredgecolor="white", func=lambda s: np.ceil(inverse_circle_area_func(s, p=p, k=k)))
    legend = ax.legend(*pubble.legend_elements(**kw), bbox_to_anchor=(0.7, 0.15), title="Parameters (Params) / M", ncol=6, fontsize=0, title_fontsize=0, handletextpad=90, frameon=False)

    ax.set(xlim=(0, 2900), ylim=(45, 90), xticks=np.arange(0, 2900, step=300), yticks=np.arange(45, 90, step=5), xlabel="Floating-point Operations Per Second (FLOPs) / GFLOPs",
           ylabel="Intersection over Union (IoU) / %")

    fig.tight_layout()
    fig.savefig("./3D_CBCT_Tooth_bubble_image.jpg", bbox_inches='tight', dpi=300)
    plt.show()


def generate_coronal_maximum_intensity_projection_image(image_path):
    # 读取图像数据
    image, spacing = utils.load_image(image_path)
    print("spacing: ", spacing)
    # 转换维度
    image = np.transpose(image, (2, 1, 0))
    image = image[::-1, :, :]
    # 最大强度投影
    coronal_MIP_image = np.max(image, axis=1)
    coronal_MIP_image = (coronal_MIP_image - coronal_MIP_image.min()) / (coronal_MIP_image.max() - coronal_MIP_image.min())
    coronal_MIP_image *= 255
    coronal_MIP_image = coronal_MIP_image.astype(np.uint8)
    # 展示结果
    plt.imshow(coronal_MIP_image, cmap="gray")
    plt.show()
    print("冠状MIP图像维度：", coronal_MIP_image.shape)
    # 保存到原始目录
    img = Image.fromarray(coronal_MIP_image)
    img.save(r"./images/12.jpg")


def generate_three_views(image_path):
    # 读取图像数据
    image, _ = utils.load_image(image_path)
    # 转换维度
    image = np.transpose(image, (2, 1, 0))
    image = image[::-1, :, :]
    image = (image - image.min()) / (image.max() - image.min())
    image *= 255
    image = image.astype(np.uint8)
    print(image.shape)
    h, d, w = image.shape
    # 获取三视图
    sagittal_plane = image[:, :, w // 2]
    coronal_plane = image[:, 90, :]
    axial_plane = image[h // 2, :, :]
    plt.imshow(sagittal_plane, cmap="gray")
    plt.show()
    plt.imshow(coronal_plane, cmap="gray")
    plt.show()
    plt.imshow(axial_plane, cmap="gray")
    plt.show()
    sagittal_plane = (sagittal_plane - sagittal_plane.min()) / (sagittal_plane.max() - sagittal_plane.min())
    sagittal_plane *= 255
    sagittal_plane = sagittal_plane.astype(np.uint8)
    coronal_plane = (coronal_plane - coronal_plane.min()) / (coronal_plane.max() - coronal_plane.min())
    coronal_plane *= 255
    coronal_plane = coronal_plane.astype(np.uint8)
    axial_plane = (axial_plane - axial_plane.min()) / (axial_plane.max() - axial_plane.min())
    axial_plane *= 255
    axial_plane = axial_plane.astype(np.uint8)
    # 调整三视图大小并拼接
    placeholder_image = np.full((224, 20), 255, dtype=np.uint8)
    sagittal_plane = cv2.resize(sagittal_plane, (int(sagittal_plane.shape[1] * 224 / sagittal_plane.shape[0]), 224))
    coronal_plane = cv2.resize(coronal_plane, (int(coronal_plane.shape[1] * 224 / coronal_plane.shape[0]), 224))
    axial_plane = cv2.resize(axial_plane, (int(axial_plane.shape[1] * 224 / axial_plane.shape[0]), 224))
    three_views_image = np.concatenate([axial_plane, placeholder_image, sagittal_plane, placeholder_image, coronal_plane], axis=1)
    # 保存三视图
    sagittal_plane_img = Image.fromarray(sagittal_plane)
    sagittal_plane_img.save(r"./images/three_views/sagittal_plane.jpg")
    coronal_plane_img = Image.fromarray(coronal_plane)
    coronal_plane_img.save(r"./images/three_views/coronal_plane.jpg")
    axial_plane_img = Image.fromarray(axial_plane)
    axial_plane_img.save(r"./images/three_views/axial_plane.jpg")
    three_views_image_img = Image.fromarray(three_views_image)
    three_views_image_img.save(r"./images/three_views/three_views_image.jpg")


def generate_surface_labels(src_root_dir):
    # 初始化目录结构
    src_labels_dir = os.path.join(src_root_dir, "labels")
    dst_labels_dir = os.path.join(src_root_dir, "surface_labels")
    if os.path.exists(dst_labels_dir):
        shutil.rmtree(dst_labels_dir)
    os.makedirs(dst_labels_dir)
    # 遍历所有labels
    for label_name in tqdm(os.listdir(src_labels_dir)):
        file_name = label_name.split(".")[0]
        # 读取
        label_path = os.path.join(src_labels_dir, label_name)
        label_np, spacing = utils.load_label(label_path, index_to_class_dict=index_to_class_dict)
        label_np = np.transpose(label_np, (2, 1, 0))
        d, w, h = label_np.shape
        label = sitk.GetImageFromArray(label_np)
        label.SetSpacing(spacing)

        # 获取标签轮廓边界点
        surface_label = sitk.LabelContour(label)
        surface_label.SetSpacing(spacing)

        # 保存
        sitk.WriteImage(surface_label, os.path.join(dst_labels_dir, file_name + ".nii.gz"))

        # plt.imshow(label_np[d//2, :, :], cmap="gray")
        # plt.show()
        # surface_label_np = sitk.GetArrayFromImage(surface_label)
        # plt.imshow(surface_label_np[d // 2, :, :], cmap="gray")
        # plt.show()
        # OrthoSlicer3D(label_np).show()
        # OrthoSlicer3D(surface_label_np).show()
        # print(np.unique(label_np))
        # print(np.unique(surface_label_np))
        # break


def generate_keypoint_heatmap_labels(src_root_dir):
    # 初始化目录结构
    src_labels_dir = os.path.join(src_root_dir, "labels")
    Centroid_labels_dir = os.path.join(src_root_dir, "centroid_labels")
    if os.path.exists(Centroid_labels_dir):
        shutil.rmtree(Centroid_labels_dir)
    os.makedirs(Centroid_labels_dir)
    # 遍历所有labels
    # for label_name in tqdm(os.listdir(src_labels_dir)):
    for label_name in tqdm(os.listdir(src_labels_dir)):
        # 划分文件名和扩展名
        file_name, ext = os.path.splitext(label_name)
        # 读取
        label_path = os.path.join(src_labels_dir, label_name)
        label_np = utils.load_image_or_label(label_path, [0.5, 0.5, 0.5], type="label", index_to_class_dict=index_to_class_dict)
        label_np = np.transpose(label_np, (2, 1, 0))
        d, w, h = label_np.shape
        label = sitk.GetImageFromArray(label_np)
        # 获取原始label中存在的所有类别标签值
        label_values = np.unique(label_np)
        # 创建统计字典
        statistic_dict = {val: None for val in range(2, 35)}
        # 由于连通区域分析只能传入二值化图像，所以每个类别单独处理
        for label_value in range(2, 35):
            # 获取当前类别的二值化图像
            binary_label_np = (label_np == label_value).astype(np.uint8)
            if np.sum(binary_label_np) == 0:  # 没有该类别直接跳过
                continue
            binary_label = sitk.GetImageFromArray(binary_label_np)
            # 连通区域分析
            cc_label = sitk.ConnectedComponent(binary_label)
            stats = sitk.LabelIntensityStatisticsImageFilter()
            stats.Execute(cc_label, binary_label)
            # 遍历所有连通区域
            for cc_label_value in stats.GetLabels():
                # 获取当前连通区域的统计数据
                area = stats.GetPhysicalSize(cc_label_value)
                centroid = stats.GetCentroid(cc_label_value)
                # 更新统计字典
                if (statistic_dict[label_value] is None) or (area > statistic_dict[label_value]["area"]):
                    statistic_dict[label_value] = {
                        "area": area,
                        "centroid": centroid
                    }
        # 初始化热力图信息
        heatmap_str = " ".join([str(val) for val in [h, w, d]]) + "\n"
        # 每个通道依次构造热力图
        for label_value in range(2, 35):
            dict_data = statistic_dict[label_value]
            if dict_data is None:
                heatmap_str += " ".join(["0", "0", "0"]) + "\n"
            else:
                heatmap_str += " ".join([str(val) for val in dict_data["centroid"]]) + "\n"
        # 保存
        txt_file_path = os.path.join(Centroid_labels_dir, file_name + ".txt")
        utils.pre_write_txt(heatmap_str, txt_file_path)


def show_dataset_spacing_resolution(root_dir):
    images_dir = os.path.join(root_dir, "images")
    for image_name in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_name)
        image, spacing = utils.load_image(image_path)
        print(image.shape, spacing)


def show_surface_labels(root_dir):
    for label_name in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label_name)
        # 读取nii对象
        NiiImage = sitk.ReadImage(label_path)
        # 从nii对象中获取numpy格式的数组，[z, y, x]
        image_numpy = sitk.GetArrayFromImage(NiiImage)
        # 转换维度为 [x, y, z]
        label_np = image_numpy.transpose(2, 1, 0)
        print(np.unique(label_np))
        OrthoSlicer3D(label_np).show()


def show_single_image(image_name, slice_index=0):
    # 初始化路径
    root_dir = r"./datasets/HX-multi-class-10"
    image_path = os.path.join(root_dir, "images", image_name + ".nrrd")
    label_path = os.path.join(root_dir, "labels", image_name + ".nrrd")
    surface_label_path = os.path.join(root_dir, "surface_labels", image_name + ".nii.gz")
    centroid_label_path = os.path.join(root_dir, "centroid_labels", image_name + ".txt")
    # 读取
    image_np = utils.load_image_or_label(image_path, [0.5, 0.5, 0.5], type="image", index_to_class_dict=index_to_class_dict)
    label_np = utils.load_image_or_label(label_path, [0.5, 0.5, 0.5], type="label", index_to_class_dict=index_to_class_dict)
    surface_label_np = utils.load_image_or_label(surface_label_path, [0.5, 0.5, 0.5], type="surface_label", index_to_class_dict=index_to_class_dict)
    centroid_label_np = utils.load_image_or_label(centroid_label_path, [0.5, 0.5, 0.5], type="centroid_label", index_to_class_dict=index_to_class_dict)
    h, w, d = label_np.shape
    centroid_label_np = np.concatenate([np.full((1, h, w, d), fill_value=0.15), centroid_label_np], axis=0)
    centroid_label_np = np.argmax(centroid_label_np, axis=0)
    print(image_np.shape)
    print(label_np.shape)
    print(surface_label_np.shape)
    print(centroid_label_np.shape)
    print(np.unique(image_np))
    print(np.unique(label_np))
    print(np.unique(surface_label_np))
    print(np.unique(centroid_label_np))
    # 获取二维切片
    image_np_2d = image_np[:, :, slice_index].transpose()
    label_np_2d = label_np[:, :, slice_index].transpose()
    surface_label_np_2d = surface_label_np[:, :, slice_index].transpose()
    centroid_label_np_2d = centroid_label_np[:, :, slice_index]
    # 裁剪ROI
    h, w = label_np_2d.shape
    y_points, x_points = np.nonzero(label_np_2d)
    x_min, x_max, y_min, y_max = x_points.min(), x_points.max(), y_points.min(), y_points.max()
    image_np_2d = image_np_2d[max(0, y_min - 10):min(h, y_max + 11), max(0, x_min - 10):min(w, x_max + 11)]
    label_np_2d = label_np_2d[max(0, y_min - 10):min(h, y_max + 11), max(0, x_min - 10):min(w, x_max + 11)]
    surface_label_np_2d = surface_label_np_2d[max(0, y_min - 10):min(h, y_max + 11), max(0, x_min - 10):min(w, x_max + 11)]
    centroid_label_np_2d = centroid_label_np_2d[max(0, y_min - 10):min(h, y_max + 11), max(0, x_min - 10):min(w, x_max + 11)]
    # 转换格式
    image_np_2d = (image_np_2d - image_np_2d.min()) / (image_np_2d.max() - image_np_2d.min())
    image_np_2d *= 255
    image_np_2d = image_np_2d.astype(np.uint8)
    label_np_2d = (label_np_2d - label_np_2d.min()) / (label_np_2d.max() - label_np_2d.min())
    label_np_2d *= 255
    label_np_2d = label_np_2d.astype(np.uint8)
    surface_label_np_2d = (surface_label_np_2d - surface_label_np_2d.min()) / (surface_label_np_2d.max() - surface_label_np_2d.min())
    surface_label_np_2d *= 255
    surface_label_np_2d = surface_label_np_2d.astype(np.uint8)
    centroid_label_np_2d = (centroid_label_np_2d - centroid_label_np_2d.min()) / (centroid_label_np_2d.max() - centroid_label_np_2d.min())
    centroid_label_np_2d *= 255
    centroid_label_np_2d = centroid_label_np_2d.astype(np.uint8)
    # 展示
    plt.imshow(image_np_2d, cmap="gray")
    plt.show()
    plt.imshow(label_np_2d, cmap="gray")
    plt.show()
    plt.imshow(surface_label_np_2d, cmap="gray")
    plt.show()
    plt.imshow(centroid_label_np_2d, cmap="gray")
    plt.show()
    # 保存
    image_img = Image.fromarray(image_np_2d)
    image_img.save(r"./images/two_stage_pictures/origin_image.jpg")
    label_img = Image.fromarray(label_np_2d)
    label_img.save(r"./images/two_stage_pictures/label.jpg")
    surface_label_img = Image.fromarray(surface_label_np_2d)
    surface_label_img.save(r"./images/two_stage_pictures/surface_label.jpg")
    centroid_label_img = Image.fromarray(centroid_label_np_2d)
    centroid_label_img.save(r"./images/two_stage_pictures/centroid_label.jpg")


if __name__ == '__main__':
    # load_nii_file(r"./datasets/NC-release-data-full/valid/labels/Teeth_0011_0000.nii.gz")

    # load_obj_file(r"./datasets/Teeth3DS/training/upper/0EAKT1CU/0EAKT1CU_upper.obj")

    # split_dataset(r"./datasets/NC-release-data-checked", train_ratio=0.8, seed=123)

    # 分析数据集中image和label的一致性和正确性
    # analyse_image_label_consistency(r"./datasets/NC-release-data")

    # 分析数据集的Clip上下界、均值和方差
    # analyse_dataset(dataset_dir=r"./datasets/HX-multi-class-10", resample_spacing=[0.5, 0.5, 0.5], clip_lower_bound_ratio=1e-6, clip_upper_bound_ratio=1 - 1e-7, classes=35)

    # 统计所有网络模型的参数量
    count_all_models_parameters(["PMFSNet", "UNet3D", "VNet", "DenseVNet", "AttentionUNet3D", "DenseVoxelNet", "MultiResUNet3D", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet"])

    # 生成牙齿数据集快照
    # generate_NC_release_data_snapshot(r"./datasets")

    # 对别两类Dice Loss
    # compare_Dice()

    # 生成牙齿数据集的样本展示图
    # generate_samples_image(scale=1)

    # 生成分割后拼接图
    # generate_segmented_sample_image(scale=1)

    # 生成气泡图
    # generate_bubble_image()

    # 生成冠状MIP图像
    # generate_coronal_maximum_intensity_projection_image(r"./temp/1001470164_20180114.nii.gz")

    # 生成三视图
    # generate_three_views(r"./datasets/NC-release-data-full/train/images/1001152328_20180112.nii.gz")

    # 生成表面轮廓标注图像数据集
    # generate_surface_labels(r"./datasets/HX-multi-class-10")

    # 生成关键点热力图标注数据集
    # generate_keypoint_heatmap_labels(r"./datasets/HX-multi-class-10")

    # 打印数据集原始体素间距和分辨率
    # show_dataset_spacing_resolution(r"./datasets/HX-multi-class-10")

    # 展示生成的表面轮廓标注
    # show_surface_labels(r"./datasets/HX-multi-class-10/surface_labels")

    # 展示某张图像的原始图像、边缘图像、标注图像
    # show_single_image("1_2", slice_index=20)
