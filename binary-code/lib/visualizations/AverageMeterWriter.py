import os
import torch
import numpy as np

from lib import utils


params = {
    # ——————————————————————————————————————————————     启动初始化    ———————————————————————————————————————————————————

    "CUDA_VISIBLE_DEVICES": "1",  # 选择可用的GPU编号

    "seed": 1777777,  # 随机种子

    "cuda": True,  # 是否使用GPU

    "benchmark": True,  # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。用于网络输入维度和类型不会变的情况。

    "deterministic": False,  # 固定cuda的随机数种子，每次返回的卷积算法将是确定的。用于复现模型结果

    # —————————————————————————————————————————————     预处理       ————————————————————————————————————————————————————

    "resample_spacing": [0.5, 0.5, 0.5],  # 重采样的体素间距。三个维度的值一般相等，可设为0.5(图像尺寸有[200,200,100]、[200,200,200]、
    # [160,160,160]),或者设为0.25(图像尺寸有[400,400,200]、[400,400,400]、[320,320,320])

    "clip_lower_bound": -1412,  # clip的下边界数值
    "clip_upper_bound": 17943,  # clip的上边界数值

    "crop_size": (160, 160, 96),  # 随机裁剪的尺寸。1、每个维度都是32的倍数这样在下采样时不会报错;2、11G的显存最大尺寸不能超过(192,192,160);
    # 3、要依据上面设置的"resample_spacing",在每个维度随机裁剪的尺寸不能超过图像重采样后的尺寸;

    # ——————————————————————————————————————————————    数据增强    ——————————————————————————————————————————————————————

    # 标准化均值
    "normalize_mean": 0.05029342141696459,
    "normalize_std": 0.028477091559295814,

    # —————————————————————————————————————————————    数据读取     ——————————————————————————————————————————————————————

    "dataset_name": "NCTooth",  # 数据集名称， 可选["NCTooth", ]

    "dataset_path": r"./datasets/NC-release-data-checked",  # 数据集路径

    "batch_size": 1,  # batch_size大小

    "num_workers": 4,  # num_workers大小

    # —————————————————————————————————————————————    网络模型     ——————————————————————————————————————————————————————

    "model_name": "PMRFNet",  # 模型名称，可选["DenseVNet","UNet3D", "VNet", "AttentionUNet", "R2UNet", "R2AttentionUNet",
    # "HighResNet3D", "DenseVoxelNet", "MultiResUNet", "PMRFNet"]

    "in_channels": 1,  # 模型最开始输入的通道数,即模态数

    "classes": 2,  # 模型最后输出的通道数,即类别总数

    "index_to_class_dict":  # 类别索引映射到类别名称的字典
    {
        0: "background",
        1: "foreground"
    },

    "pretrain": None,  # 是否需要加载预训练权重，如果需要则指定预训练权重文件路径

    # ————————————————————————————————————————————    损失函数     ———————————————————————————————————————————————————————

    "metric_names": ["DSC", "ASSD", "HD"],  # 采用除了dsc之外的评价指标，可选["DSC", "ASSD", "HD", "SO"]

    "sigmoid_normalization": False,  # 对网络输出的各通道进行归一化的方式,True是对各元素进行sigmoid,False是对所有通道进行softmax

    "dice_mode": "standard",  # DSC的计算方式，"standard":标准计算方式；"extension":扩展计算方式

    # ————————————————————————————————————————————   测试相关参数   ———————————————————————————————————————————————————————

    "crop_stride": [32, 32, 32],  # 验证或者测试时滑动分割的移动步长

    "test_type": 2,  # 测试类型，0：单张图像无标签；1：单张图像有标签；2：测试集批量测试

    "single_image_path": None,  # 单张图像的路径

    "single_label_path": None,  # 单张标注图像的路径

}


class AverageMeterWriter(object):
    """ Computes and stores the average and current value """
    def __init__(self, opt):
        self.opt = opt
        self.class_names = list(self.opt["index_to_class_dict"].values())
        self.statistics_dict = self.init()
        self.reset()

    def init(self):
        # 初始化所有评价指标在所有类别上的统计数据
        statistics_dict = {
            metric_name: {class_name: 0.0 for class_name in self.class_names}
            for metric_name in self.opt["metric_names"]
        }
        # 初始化所有评价指标在所有类别上的平均值
        for metric_name in self.opt["metric_names"]:
            statistics_dict[metric_name]["avg"] = 0.0
        # 初始化各类别计数
        statistics_dict["class_count"] = {class_name: 0 for class_name in self.class_names}
        # 初始化所有样本计数
        statistics_dict["count"] = 0

        return statistics_dict

    def reset(self):
        # 重置所有样本计数
        self.statistics_dict["count"] = 0
        # 重置各类别计数
        for class_name in self.class_names:
            self.statistics_dict["class_count"][class_name] = 0
        # 重置平均评价指标
        for metric_name in self.opt["metric_names"]:
            self.statistics_dict[metric_name]["avg"] = 0.0
            # 重置各类别的各评价指标
            for class_name in self.class_names:
                self.statistics_dict[metric_name][class_name] = 0.0

    def update(self, per_class_metrics, target, cur_batch_size):
        """
        更新统计字典
        :param per_class_metrics: 所有计算得到的评价指标列表
        :param target: 标注图像
        :param cur_batch_size: batch大小
        :return:
        """
        # 计算出现的类别mask
        mask = torch.zeros(self.opt["classes"])
        unique_index = torch.unique(target).int()
        for index in unique_index:
            mask[index] = 1
        # 更新总样本数
        self.statistics_dict["count"] += cur_batch_size
        # 更新各类别计数
        for i, class_name in enumerate(self.class_names):
            if mask[i] == 1:
                self.statistics_dict["class_count"][class_name] += cur_batch_size
        # 更新各评价指标
        for i, metric_name in enumerate(self.opt["metric_names"]):
            # 获取当前评价指标在各类别上的数值
            per_class_metric = per_class_metrics[i]
            # 只需要mask部分
            per_class_metric = per_class_metric * mask
            # 更新平均评价指标
            self.statistics_dict[metric_name]["avg"] += (torch.sum(per_class_metric) / torch.sum(
                mask)).item() * cur_batch_size
            # 更新各类别的各评价指标
            for j, class_name in enumerate(self.class_names):
                self.statistics_dict[metric_name][class_name] += per_class_metric[j].item() * cur_batch_size

    def display_terminal(self):
        print_info = ""
        # 评价指标作为列名
        print_info += " " * 12
        for metric_name in self.opt["metric_names"]:
            print_info += "{:^12}".format(metric_name)
        print_info += '\n'
        # 按每行依次添加各类别的所有评价指标
        for class_name in self.class_names:
            print_info += "{:<12}".format(class_name)
            for metric_name in self.opt["metric_names"]:
                value = 0
                if self.statistics_dict["class_count"][class_name] != 0:
                    value = self.statistics_dict[metric_name][class_name] / self.statistics_dict["class_count"][class_name]
                print_info += "{:^12.6f}".format(value)
            print_info += '\n'
        # 添加最后一行为各评价指标在各类别上的均值
        print_info += "{:<12}".format("average")
        for metric_name in self.opt["metric_names"]:
            value = 0
            if self.statistics_dict["count"] != 0:
                value = self.statistics_dict[metric_name]["avg"] / self.statistics_dict["count"]
            print_info += "{:^12.6f}".format(value)
        print(print_info)
        result_txt_path = os.path.join(self.opt["run_dir"], utils.datestr() + "_" + self.opt["model_name"] + ".txt")
        utils.pre_write_txt(print_info, result_txt_path)




if __name__ == '__main__':
    writer = AverageMeterWriter(params)
    writer.display_terminal()










