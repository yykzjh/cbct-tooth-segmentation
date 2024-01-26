# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/4/19 16:28
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import glob
import numpy as np

from torch.utils.data import Dataset

import lib.transforms as transforms
import lib.utils as utils


class BinaryToothDataset(Dataset):
    """
    读取二分类牙齿数据集
    """

    def __init__(self, opt, mode):
        """
        Args:
            opt: 参数字典
            mode: train/valid
        """
        self.opt = opt
        self.mode = mode
        self.root = opt["dataset_path"]
        self.train_path = os.path.join(self.root, "train")
        self.val_path = os.path.join(self.root, "valid")

        # 分类创建子卷数据集
        if self.mode == 'train':
            # 初始化数据增强列表
            self.augmentations = [
                opt["open_elastic_transform"], opt["open_gaussian_noise"], opt["open_random_flip"],
                opt["open_random_rescale"], opt["open_random_rotate"], opt["open_random_shift"]]
            # 初始化子卷数据集目录
            self.sub_volume_root_dir = os.path.join(self.root, "sub_volumes")
            if not os.path.exists(self.sub_volume_root_dir):
                os.makedirs(self.sub_volume_root_dir)
            self.sub_volume_path = os.path.join(self.sub_volume_root_dir,
                                                "-".join([str(item) for item in opt["crop_size"]])
                                                + "_" + str(opt["samples_train"])
                                                + "_" + str(opt["crop_threshold"]) + ".npz")
            # 初始化子卷数据集存储数据结构
            self.selected_images = []
            self.selected_position = []
            # 定义数据增强
            all_augments = [
                 transforms.ElasticTransform(alpha=opt["elastic_transform_alpha"],
                                             sigma=opt["elastic_transform_sigma"]),
                 transforms.GaussianNoise(mean=opt["gaussian_noise_mean"],
                                          std=opt["gaussian_noise_std"]),
                 transforms.RandomFlip(),
                 transforms.RandomRescale(min_percentage=opt["random_rescale_min_percentage"],
                                          max_percentage=opt["random_rescale_max_percentage"]),
                 transforms.RandomRotation(min_angle=opt["random_rotate_min_angle"],
                                           max_angle=opt["random_rotate_max_angle"]),
                 transforms.RandomShift(max_percentage=opt["random_shift_max_percentage"])
            ]
            # 获取实际要进行的数据增强
            practice_augments = [all_augments[i] for i, is_open in enumerate(self.augmentations) if is_open]
            # 定义数据增强方式
            if opt["augmentation_method"] == "Choice":
                self.train_transforms = transforms.ComposeTransforms([
                    transforms.ClipAndShift(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                    transforms.RandomAugmentChoice(practice_augments, p=opt["augmentation_probability"]),
                    transforms.ToTensor(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                    transforms.Normalize(opt["normalize_mean"], opt["normalize_std"])
                ])
            elif opt["augmentation_method"] == "Compose":
                self.train_transforms = transforms.ComposeTransforms([
                    transforms.ClipAndShift(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                    transforms.RandomAugmentCompose(practice_augments, p=opt["augmentation_probability"]),
                    transforms.ToTensor(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                    transforms.Normalize(opt["normalize_mean"], opt["normalize_std"])
                ])

            # 判断需不需要重新分割子数据集用于训练
            if (not opt["create_data"]) and os.path.isfile(self.sub_volume_path):
                # 读取子卷数据集的信息
                sub_volume_dict = np.load(self.sub_volume_path)
                self.selected_images = [tuple(image) for image in sub_volume_dict["selected_images"]]
                self.selected_position = [tuple(crop_point) for crop_point in sub_volume_dict["selected_position"]]
            else:  # 如果需要创建子数据集，或者没有存储子数据集信息的文件
                # 获取数据集中所有原图图像和标注图像的路径
                images_path_list = sorted(glob.glob(os.path.join(self.train_path, "images", "*.nii.gz")))
                labels_path_list = sorted(glob.glob(os.path.join(self.train_path, "labels", "*.nii.gz")))

                # 生成子卷数据集
                self.selected_images, self.selected_position = utils.create_sub_volumes(images_path_list, labels_path_list, opt)

                # 保存子卷数据集信息
                np.savez(self.sub_volume_path, selected_images=self.selected_images, selected_position=self.selected_position)

        elif self.mode == 'valid':
            # 定义验证集数据增强
            self.val_transforms = transforms.ComposeTransforms([
                transforms.ClipAndShift(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                transforms.ToTensor(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                transforms.Normalize(opt["normalize_mean"], opt["normalize_std"])
            ])

            # 获取数据集中所有原图图像和标注图像的路径
            images_path_list = sorted(glob.glob(os.path.join(self.val_path, "images", "*.nii.gz")))
            labels_path_list = sorted(glob.glob(os.path.join(self.val_path, "labels", "*.nii.gz")))

            # 得到验证集数据
            self.selected_images = list(zip(images_path_list, labels_path_list))


    def __len__(self):
        return len(self.selected_images)


    def __getitem__(self, index):
        # 先获取原图图像和标注图像的路径
        image_path, label_path = self.selected_images[index]
        # 读取原始图像和标注图像
        image_np = utils.load_image_or_label(image_path, self.opt["resample_spacing"], type="image")
        label_np = utils.load_image_or_label(label_path, self.opt["resample_spacing"], type="label")

        if self.mode == 'train':  # 训练集
            # 获取随机裁剪的位置
            crop_point = self.selected_position[index]
            # 随机裁剪
            crop_image_np = utils.crop_img(image_np, self.opt["crop_size"], crop_point)
            crop_label_np = utils.crop_img(label_np, self.opt["crop_size"], crop_point)
            # 数据变换和数据增强
            transform_image, transform_label = self.train_transforms(crop_image_np, crop_label_np)
            return transform_image.unsqueeze(0), transform_label

        else:  # 验证集
            # 数据变换和数据增强
            transform_image, transform_label = self.val_transforms(image_np, label_np)
            return transform_image.unsqueeze(0), transform_label
