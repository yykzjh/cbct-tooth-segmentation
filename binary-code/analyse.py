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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import proplot as pplt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import tensorrt as trt

import lib.utils as utils
import lib.models as models


def calc_grayhist(image, max_length=255):
    # 无论多少维数组都展成一维
    image_flatten = image.flatten()
    image_flatten = np.array(image_flatten, dtype=np.int32)
    # plt.hist(image_flatten, bins=max_length+1, color="g", histtype="bar", rwidth=1, alpha=0.6)
    # plt.show()
    # print(image_flatten.shape[0])
    # 求出像素值0~max_length的统计直方图
    grayhist = np.bincount(image_flatten)
    # 归一化直方图
    grayhist = grayhist / image_flatten.shape[0]
    return grayhist


def load_nii_file(file_path):
    """
    底层读取.nii.gz文件
    Args:
        file_path: 文件路径
        type: image或者label

    Returns: uint16格式的三维numpy数组, spacing三个元素的元组

    """
    # 读取图像数据
    image_np, spacing = utils.load_image(file_path)
    print("spacing: ", spacing)
    # 转换维度
    image_np = np.transpose(image_np, (2, 1, 0))
    image_np = image_np[::-1, :, :]
    image_np = image_np - image_np.min()
    print(image_np.min(), image_np.max())

    # 显示直方图
    plt.hist(image_np.flatten(), bins=1000, range=(1, 10000))
    plt.show()

    T = 2880
    mask = np.zeros_like(image_np).astype(np.uint8)
    mask[image_np > T] = 255
    OrthoSlicer3D(mask).show()
    coronal_MIP_image = np.max(mask, axis=1)
    plt.imshow(coronal_MIP_image, cmap="gray")
    plt.show()

    # 获取3D图像的高斯分布阈值
    hist = calc_grayhist(image_np)
    # 把像素值为0的频率去掉
    hist[0] = 0
    print("3D图像分割阈值：", T)
    plt.hist(image_np.flatten(), bins=1000, range=(1, 10000), color="g", histtype="bar", rwidth=1, alpha=0.6)
    plt.axvline(T)
    plt.show()

    # # 最大强度投影
    # coronal_MIP_image = np.max(image_np, axis=1)
    # coronal_MIP_image = coronal_MIP_image - coronal_MIP_image.min()
    # print(coronal_MIP_image.min(), coronal_MIP_image.max())
    # # 展示结果
    # plt.imshow(coronal_MIP_image, cmap="gray")
    # plt.show()
    #
    # # 显示直方图
    # plt.hist(coronal_MIP_image.flatten(), bins=1000, range=(1, 7000))
    # plt.show()
    #
    # T = 1260
    # mask = np.zeros_like(coronal_MIP_image).astype(np.uint8)
    # mask[coronal_MIP_image > T] = 255
    # plt.imshow(mask, cmap="gray")
    # plt.show()


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
    images_list = sorted(glob.glob(os.path.join(dataset_dir, "*", "images", "*.nii.gz")))
    # 加载数据集中所有标注图像
    labels_list = sorted(glob.glob(os.path.join(dataset_dir, "*", "labels", "*.nii.gz")))
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
        label_np = utils.load_image_or_label(labels_list[i], resample_spacing, type="label")
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
        label_np = utils.load_image_or_label(labels_list[i], resample_spacing, type="label")
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
        "classes": 2,
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
    # 创建整个大图
    image = np.full((960, 1295, 3), 255)
    # 依次遍历
    for i in range(4):
        for j in range(4):
            x = i * (224 + 8)
            y = j * (320 + 5)
            # 读取并处理图像
            img = cv2.imread(r"./images/challenging_samples/" + str(i) + str(j) + ".jpg")
            img = cv2.resize(img, (320, 224))
            image[x: x + 224, y: y + 320, :] = img
    image = image[:, :, ::-1]
    # 添加文字
    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 36)
    color = (0, 0, 0)

    position1 = (60, 917)
    text1 = "Original Image"
    draw.text(position1, text1, font=font, fill=color)

    position2 = (350, 917)
    text2 = "Ground Truth (2D)"
    draw.text(position2, text2, font=font, fill=color)

    position3 = (665, 917)
    text3 = "Coronal MIP Image"
    draw.text(position3, text3, font=font, fill=color)

    position4 = (990, 917)
    text4 = "Ground Truth (3D)"
    draw.text(position4, text4, font=font, fill=color)

    image.show()
    w, h = image.size
    image = image.resize((scale * w, scale * h), resample=Image.Resampling.BILINEAR)
    print(image.size)
    image.save(r"./images/challenging_samples/challenging_samples.jpg")


def generate_segmented_sample_image(scale=1):
    # 设置选择的图像编号
    selected_images = [12, 21, 31]
    # 设置几列
    cols = 4
    # 设置一些参数
    span1 = 234
    span2 = 330
    span3 = 752
    # 创建整个大图
    image = np.full((2256, 1310, 3), 255)
    # 依次遍历
    for i in range(3):
        x_min, x_max, y_min, y_max = 0, 0, 0, 0
        for j in range(12):
            # 获取图像
            img = cv2.imread(os.path.join(r"./images/NC-release-data_segment_result_samples", "{:04d}_{:02d}".format(selected_images[i], j) + ".jpg"))
            # 旋转图像
            img = np.rot90(img, -1)
            # 判断如果是ground truth，则计算裁剪区域
            if j == 0:
                h, w, _ = img.shape
                y_points, x_points, _ = np.nonzero(img)
                x_min, x_max, y_min, y_max = x_points.min(), x_points.max(), y_points.min(), y_points.max()
            # 裁剪图像
            img = img[max(0, y_min - 10):min(h, y_max + 11), max(0, x_min - 10):min(w, x_max + 11), :]
            # 缩放图像
            img = cv2.resize(img, (320, 224))
            # 计算拼接位置
            pos_y, pos_x = (j // cols) * span3 + i * span1, (j % cols) * span2
            # 拼接
            image[pos_y: pos_y + 224, pos_x: pos_x + 320, :] = img
    image = image[:, :, ::-1]

    # 添加文字的设置
    col_names = ["Ground Truth", "UNet3D", "DenseVNet", "AttentionUNet3D", "DenseVoxelNet", "MultiResUNet3D", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet", "PMFSNet"]
    col_positions = [60, 430, 735, 1020, 40, 365, 760, 1055, 80, 420, 745, 1075]

    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 36)
    color = (0, 0, 0)

    # 遍历添加文字
    for i, text in enumerate(col_names):
        # 计算位置
        position_x, position_y = col_positions[i], (i // cols) * span3 + 698
        # 添加文字
        draw.text((position_x, position_y), text, font=font, fill=color)

    image.show()
    w, h = image.size
    image = image.resize((scale * w, scale * h), resample=Image.Resampling.BILINEAR)
    image.save(r"./One_Binary_Compare_Visualization.jpg")


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
    values = data.DSC
    xtext_positions = [2250, 20, 2300, 1250, 225, 1010, 200, 600, 1810, 400, 10]
    ytext_positions = [64.4, 93, 83.4, 79.1, 86.5, 93.7, 68.9, 95.7, 91.0, 91.2, 96.9]
    legend_sizes = [1, 5, 25, 50, 100, 150]
    legend_yposition = 73
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

    ax.set(xlim=(0, 2900), ylim=(60, 105), xticks=np.arange(0, 2900, step=300), yticks=np.arange(60, 105, step=5), xlabel="Floating-point Operations Per Second (FLOPs) / GFLOPs",
           ylabel="Dice Similarity Coefficient (DSC) / %")

    fig.tight_layout()
    fig.savefig("./One_Binary_Compare_Bubble.jpg", bbox_inches='tight', dpi=300)
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
        # 读取
        label_path = os.path.join(src_labels_dir, label_name)
        label = sitk.ReadImage(label_path)
        spacing = label.GetSpacing()
        direction = label.GetDirection()
        origin = label.GetOrigin()
        label_np = sitk.GetArrayFromImage(label)
        d, w, h = label_np.shape
        label_np[label_np > 0] = 1
        preprocessed_label = sitk.GetImageFromArray(label_np)
        preprocessed_label.SetSpacing(spacing)
        preprocessed_label.SetDirection(direction)
        preprocessed_label.SetOrigin(origin)

        # 获取变脸轮廓边界点
        surface_label = sitk.LabelContour(preprocessed_label)
        surface_label.SetSpacing(spacing)
        surface_label.SetDirection(direction)
        surface_label.SetOrigin(origin)

        # 保存
        sitk.WriteImage(surface_label, os.path.join(dst_labels_dir, label_name))

        # plt.imshow(label_np[d//2, :, :], cmap="gray")
        # plt.show()
        # surface_label_np = sitk.GetArrayFromImage(surface_label)
        # plt.imshow(surface_label_np[d // 2, :, :], cmap="gray")
        # plt.show()
        # break


def show_valid_set(root_dir):
    label_dir = os.path.join(root_dir, "labels")
    for label_name in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_name)
        label_np = utils.load_image_or_label(label_path, [0.5, 0.5, 0.5], type="label")
        print(label_name, ": ", label_np.shape)


def show_single_image(image_path, slice_index=0):
    label_path = image_path.replace("images", "labels")
    surface_label_path = image_path.replace("images", "surface_labels")
    image_np = utils.load_image_or_label(image_path, [0.5, 0.5, 0.5], type="image")
    label_np = utils.load_image_or_label(label_path, [0.5, 0.5, 0.5], type="label")
    surface_label_np = utils.load_image_or_label(surface_label_path, [0.5, 0.5, 0.5], type="label")
    image_np_2d = image_np[:, :, slice_index].transpose()
    label_np_2d = label_np[:, :, slice_index].transpose()
    surface_label_np_2d = surface_label_np[:, :, slice_index].transpose()
    # 裁剪ROI
    h, w = label_np_2d.shape
    y_points, x_points = np.nonzero(label_np_2d)
    x_min, x_max, y_min, y_max = x_points.min(), x_points.max(), y_points.min(), y_points.max()
    image_np_2d = image_np_2d[max(0, y_min - 10):min(h, y_max + 11), max(0, x_min - 10):min(w, x_max + 11)]
    label_np_2d = label_np_2d[max(0, y_min - 10):min(h, y_max + 11), max(0, x_min - 10):min(w, x_max + 11)]
    surface_label_np_2d = surface_label_np_2d[max(0, y_min - 10):min(h, y_max + 11), max(0, x_min - 10):min(w, x_max + 11)]
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
    # 展示
    plt.imshow(image_np_2d, cmap="gray")
    plt.show()
    plt.imshow(label_np_2d, cmap="gray")
    plt.show()
    plt.imshow(surface_label_np_2d, cmap="gray")
    plt.show()
    # 保存
    image_img = Image.fromarray(image_np_2d)
    image_img.save(r"./images/two_stage_pictures/origin_image.jpg")
    label_img = Image.fromarray(label_np_2d)
    label_img.save(r"./images/two_stage_pictures/label.jpg")
    surface_label_img = Image.fromarray(surface_label_np_2d)
    surface_label_img.save(r"./images/two_stage_pictures/surface_label.jpg")


def generate_dsc_plot_image(model_names):
    rc["font.family"] = "Times New Roman"
    rc["axes.labelsize"] = 24
    rc["tick.labelsize"] = 16
    rc["suptitle.size"] = 11
    rc["title.size"] = 11

    # 设置可选记号表
    cand_markers = ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's', 'D', 'd', 'p', 'H', 'h', 'v', '^', '<', '>', '1', '2', '3', '4', '|', '_']
    # 初始化数据存储
    xdata = []
    ydata = {}
    # 遍历所有模型
    for model_name in model_names:
        # 获取日志文件路径
        log_txt_path = os.path.join(r"./logs", model_name + ".txt")
        # 初始化纵坐标
        ydata[model_name] = [0]
        # 打开日志文件
        with open(log_txt_path, "r") as file:
            line_num = 1
            for line in file:
                # 跳过epoch参数
                if line_num % 9 != 0:
                    line_num += 1
                    continue
                line_num += 1
                line = line.strip()
                if "valid_dsc:" in line:
                    # 找到 "valid_dsc:" 的位置
                    dsc_start = line.find("valid_dsc:")
                    if dsc_start != -1:
                        # 提取 "valid_dsc:" 后面的数字
                        dsc_str = line[dsc_start + len("valid_dsc:"): dsc_start + len("valid_dsc:") + 8].strip()
                        try:
                            dsc_value = float(dsc_str)
                            dsc_value = round(dsc_value * 100, 4)
                            ydata[model_name].append(dsc_value)
                        except ValueError:
                            print("字符串转浮点数出错")
    xdata = [i for i in range(len(ydata[model_names[0]]))]

    # 画图
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, model_name in enumerate(model_names):
        ax.plot(xdata, ydata[model_name], label=model_name, marker=cand_markers[i])
    ax.legend(fontsize=14)
    plt.xticks(range(0, 21), labels=[str(i) for i in range(0, 21)])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Similarity Coefficient (DSC) / %")
    plt.savefig(r"./One_Binary_Compare_Optimize.jpg", dpi=300)
    plt.show()


def generate_experience_data(origin_excel_path, output_fle_name="test", scale_lower_bound=0.1, scale_upper_bound=2, bias=0.45, noise_lower_bound=-0.4, noise_upper_bound=0.4, gen_std=False):
    # 设置显示的小数点位数
    pd.options.display.precision = 8
    # 读取Excel文件特定范围内的数据
    df = pd.read_excel(origin_excel_path, sheet_name=0, header=0, usecols="F:J", nrows=11, dtype=str)
    df = df.astype("float64")
    # 划分两个部分
    df1 = df.iloc[:, :2]
    df2 = df.iloc[:, 2:]

    # 处理前一个部分
    scale_ratio = np.random.uniform(scale_lower_bound, scale_upper_bound, size=df1.shape)
    df1 = df1 * scale_ratio

    # 处理后一个部分
    # 整体增加偏移
    df2 += bias
    # 给每个数值加上随机噪声
    noise = np.random.uniform(noise_lower_bound, noise_upper_bound, size=df2.shape)
    df2 += noise

    # 拼在一起
    df = pd.concat([df1, df2], axis=1)

    # 多类别添加std
    if gen_std:
        std = pd.DataFrame(np.random.uniform(0.002, 0.008, size=df.shape))
        df = pd.concat([df, std], axis=0)

    # 保存到新的Excel文件中
    output_path = os.path.join(r"./docs", output_fle_name + ".xlsx")
    df.to_excel(output_path, index=False)


class TRTUtil(object):
    def __init__(self, onnx_file_path=None, engine_file_path=None, precision_flop=None, img_size=(640, 640)) -> None:
        self.logger = trt.Logger(trt.Logger.WARNING)

    def onnx_to_TRT_model(self, onnx_file_path, engine_file_path, precision_flop):
        """构建期 -> 转换网络模型为 TRT 模型

        Args:
            onnx_file_path  : 要转换的 onnx 模型的路径
            engine_file_path: 转换之后的 TRT engine 的路径
            precision_flop  : 转换过程中所使用的精度

        Returns:
            转化成功: engine
            转换失败: None
        """
        # ---------------------------------#
        # 准备全局信息
        # ---------------------------------#
        # 构建一个 构建器
        builder = trt.Builder(self.logger)
        builder.max_batch_size = 1

        # ---------------------------------#
        # 第一步，读取 onnx
        # ---------------------------------#
        # 1-1、设置网络读取的 flag
        # EXPLICIT_BATCH 相教于 IMPLICIT_BATCH 模式，会显示的将 batch 的维度包含在张量维度当中，
        # 有了 batch大小的，我们就可以进行一些必须包含 batch 大小的操作了，如 Layer Normalization。
        # 不然在推理阶段，应当指定推理的 batch 的大小。目前主流的使用的 EXPLICIT_BATCH 模式
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        # 1-3、构建一个空的网络计算图
        network = builder.create_network(network_flags)
        # 1-4、将空的网络计算图和相应的 logger 设置装载进一个 解析器里面
        parser = trt.OnnxParser(network, self.logger)
        # 1-5、打开 onnx 压缩文件，进行模型的解析工作。
        # 解析器 工作完成之后，网络计算图的内容为我们所解析的网络的内容。
        onnx_file_path = os.path.realpath(onnx_file_path)  # 将路径转换为绝对路径防止出错
        if not os.path.isfile(onnx_file_path):
            print("ONNX file not exist. Please check the onnx file path is right ? ")
            return None
        else:
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the onnx file {} . ".format(onnx_file_path))
                    # 出错了，将相关错误的地方打印出来，进行可视化处理`-`
                    for error in range(parser.num_errors):
                        print(parser.num_errors)
                        print(parser.get_error(error))
                    return None
            print("Completed parsing ONNX file . ")
        # 6、将转换之后的模型的输入输出的对应的大小进行打印，从而进行验证
        for i in range(network.num_outputs):
            print(i)
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        print("Network Description")
        batch_size = 0
        for inp in inputs:
            # 获取当前转化之前的 输入的 batch_size
            batch_size = inp.shape[0]
            print("Input '{}' with shape {} and dtype {} . ".format(inp.name, inp.shape, inp.dtype))
        for outp in outputs:
            print("Output '{}' with shape {} and dtype {} . ".format(outp.name, outp.shape, outp.dtype))
        # 确保 输入的 batch_size 不为零
        # assert batch_size > 0, "输入的 batch_size < 0, 请确定输入的参数是否满足要求. "

        # ---------------------------------#
        # 第二步，转换为 TRT 模型
        # ---------------------------------#
        # 2-1、设置 构建器 的 相关配置器
        # 应当丢弃老版本的 builder. 进行设置的操作
        config = builder.create_builder_config()
        # 2-2、设置 可以为 TensorRT 提供策略的策略源。如CUBLAS、CUDNN 等
        # 也就是在矩阵计算和内存拷贝的过程中选择不同的策略
        # config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
        # 2-3、给出模型中任一层能使用的内存上限，这里是 2^30,为 2GB
        # 每一层需要多少内存系统分配多少，并不是每次都分 2 GB
        config.max_workspace_size = 1 << 30
        # 2-4、设置 模型的转化精度
        if precision_flop == "FP32":
            # config.set_flag(trt.BuilderFlag.FP32)
            pass
        elif precision_flop == "FP16":
            if not builder.platform_has_fast_fp16:
                print("FP16 is not supported natively on this platform/device . ")
            else:
                config.set_flag(trt.BuilderFlag.FP16)
        elif precision_flop == "INT8":
            config.set_flag(trt.BuilderFlag.INT8)
        # 2-5，从构建器 构建引擎
        engine = builder.build_engine(network, config)

        # ---------------------------------#
        # 第三步，生成 SerializedNetwork
        # ---------------------------------#
        # 3-1、删除已经已经存在的版本
        engine_file_path = os.path.realpath(engine_file_path)  # 将路径转换为绝对路径防止出错
        if os.path.isfile(engine_file_path):
            try:
                os.remove(engine_file_path)
            except Exception:
                print("Cannot removing existing file: {} ".format(engine_file_path))
        print("Creating Tensorrt Engine: {}".format(engine_file_path))
        # 3-2、打开要写入的 TRT engine，利用引擎写入
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print("ONNX -> TRT Success。 Serialized Engine Saved at: {} . ".format(engine_file_path))

        return engine



def quantize_models(model_names):
    # # 参数字典
    # opt = {
    #     "in_channels": 1,
    #     "classes": 2,
    #     "device": "cuda:0",
    #     "scaling_version": "TINY",
    #     "with_pmfs_block": False,
    #     "two_stage": False,
    #     "surface_pretrain": None,
    #     "centroid_pretrain": None
    # }
    #
    # for model_name in model_names:
    #     print(model_name)
    #     # 一些参数
    #     input_size = (160, 160, 96)
    #     onnx_dir = r"./weights/onnx"
    #     split_model_name = model_name.split("-")
    #     if len(split_model_name) == 2:
    #         opt["model_name"] = split_model_name[0]
    #         opt["scaling_version"] = split_model_name[1]
    #         model = models.get_model(opt)
    #     else:
    #         opt["model_name"] = split_model_name[0]
    #         model = models.get_model(opt)
    #
    #     dummy_input = torch.randn(1, 1, input_size[0], input_size[1], input_size[2]).to("cuda")
    #     model.eval()
    #     model(dummy_input)
    #     im = torch.zeros(1, 1, input_size[0], input_size[1], input_size[2]).to("cuda")
    #     torch.onnx.export(model, im,
    #                       os.path.join(onnx_dir, model_name + ".onnx"),
    #                       verbose=False,
    #                       opset_version=12,
    #                       training=torch.onnx.TrainingMode.EVAL,
    #                       do_constant_folding=True,
    #                       input_names=['input'],
    #                       output_names=['output']
    #                       )


    trt_util = TRTUtil()
    for model_name in model_names:
        print(model_name)
        # 一些参数
        input_size = (160, 160, 96)
        onnx_dir = r"./weights/onnx"
        engine_dir = r"./weights/engine"
        trt_util.onnx_to_TRT_model(
            os.path.join(onnx_dir, model_name + ".onnx"),
            os.path.join(engine_dir, model_name + ".engine"),
            "FP16"
        )





if __name__ == '__main__':
    # load_nii_file(r"./datasets/NC-release-data-full/train/images/1001484858_20150118.nii.gz")

    # load_obj_file(r"./datasets/Teeth3DS/training/upper/0EAKT1CU/0EAKT1CU_upper.obj")

    # split_dataset(r"./datasets/NC-release-data-checked", train_ratio=0.8, seed=123)

    # 分析数据集中image和label的一致性和正确性
    # analyse_image_label_consistency(r"./datasets/NC-release-data")

    # 分析数据集的Clip上下界、均值和方差
    # analyse_dataset(dataset_dir=r"./datasets/NC-release-data-full", resample_spacing=[0.5, 0.5, 0.5], clip_lower_bound_ratio=1e-6, clip_upper_bound_ratio=1-1e-7)

    # 统计所有网络模型的参数量
    # count_all_models_parameters(["PMFSNet", "UNet3D", "DenseVNet", "AttentionUNet3D", "DenseVoxelNet", "MultiResUNet3D", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet"])

    # 生成牙齿数据集快照
    # generate_NC_release_data_snapshot(r"./datasets")

    # 对别两类Dice Loss
    # compare_Dice()

    # 生成牙齿数据集的样本展示图
    # generate_samples_image(scale=1)

    # 生成分割后拼接图
    generate_segmented_sample_image(scale=1)

    # 生成气泡图
    # generate_bubble_image()

    # 生成冠状MIP图像
    # generate_coronal_maximum_intensity_projection_image(r"./temp/1001470164_20180114.nii.gz")

    # 生成三视图
    # generate_three_views(r"./datasets/NC-release-data-full/train/images/1001152328_20180112.nii.gz")

    # 生成表面轮廓标注图像数据集
    # generate_surface_labels(r"./datasets/NC-release-data-full/valid")

    # 打印验证集所有图像尺寸
    # show_valid_set(r"./datasets/NC-release-data-full/valid")

    # 展示某张图像的原始图像、边缘图像、标注图像
    # show_single_image(r"./datasets/NC-release-data-full/train/images/1000889125_20200421.nii.gz", slice_index=100)

    # 根据日志文件生成几个模型训练过程的dsc折线图
    # generate_dsc_plot_image(["PMFSNet-TINY", "PMFSNet-SMALL", "PMFSNet-BASIC", "UNet3D", "DenseVNet", "AttentionUNet3D", "DenseVoxelNet", "MultiResUNet3D", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet"])

    # 生成实验数据
    # generate_experience_data(r"./docs/所有实验数据表格.xlsx", output_fle_name="two_multi_ablation", scale_lower_bound=1.1, scale_upper_bound=1.4, bias=-4, noise_lower_bound=-0.5, noise_upper_bound=0.5, gen_std=True)

    # 量化所有模型
    # quantize_models(["PMFSNet-TINY", "PMFSNet-SMALL", "PMFSNet-BASIC", "UNet3D", "DenseVNet", "AttentionUNet3D", "DenseVoxelNet", "MultiResUNet3D", "UNETR"])

