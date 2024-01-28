# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/27 20:14
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import numpy as np
import math
from collections import Counter

from lib import utils

import matplotlib.pyplot as plt
from mayavi import mlab




# 展示预测分割图和标注图的分布直方图对比
def display_compare_hist(label1, label2):
    # 获取类别标签个数
    bins = config.classes
    # 设置解决中文乱码问题
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 将数据扁平化
    label1_flatten = label1.flatten()
    label2_flatten = label2.flatten()

    # 初始化统计数据
    count1 = np.zeros((bins, ))
    count2 = np.zeros((bins, ))

    # 统计标签数值
    label1_count = Counter(label1_flatten)
    label2_count = Counter(label2_flatten)

    # 赋值给统计数据
    for num, cnt in label1_count.items():
        num = int(num)
        if num != 0:
            count1[num] = cnt
    for num, cnt in label2_count.items():
        num = int(num)
        if num != 0:
            count2[num] = cnt

    # 定义柱状的宽度
    width = 0.3

    # 画图
    plt.bar([i - width / 2 for i in range(bins)], count1, width=width, label='Label')
    plt.bar([i + width / 2 for i in range(bins)], count2, width=width, label='Pred')

    # 设置图例
    plt.legend()
    # 设置标题
    plt.title('标签图和预测图柱状分布对比')
    # 设置轴上的标题
    plt.xlabel("类别标签值")
    plt.ylabel("统计数量")
    # 设置x轴注释
    plt.xticks([i for i in range(bins)], list(range(bins)))

    plt.show()





# 分割图的3D可视化
def display_segmentation_3D(class_map):
    # 读取索引文件
    index_to_class_dict = utils.load_json_file(
        os.path.join(r"./lib/dataloaders/index_to_class", config.dataset_name + ".json"))
    class_to_index_dict = {}
    # 获得类别到索引的映射字典
    for key, val in index_to_class_dict.items():
        class_to_index_dict[val] = key

    # 初始化三个轴的坐标列表
    x = []
    y = []
    z = []
    # 初始化每个点对应的类别索引列表
    label = []
    # 遍历所有类别的索引
    for index in [int(index) for index in index_to_class_dict.keys()]:
        # 获取值为当前类别索引的所有点的坐标
        pos_x, pos_y, pos_z = np.nonzero(class_map == index)
        x.extend(list(pos_x))
        y.extend(list(pos_y))
        z.extend(list(pos_z))
        # 添加和点数相同数量的类别索引
        label.extend([index] * len(pos_x))

    # 自定义每个牙齿类别的颜色
    color_table = np.array([
        [255, 255, 255, 0],  # 0 background
        [255, 255, 255, 30],  # 1 gum
        [255, 215, 0, 255],  # 2 implant
        [85, 0, 0, 255],  # 3 ul1
        [255, 0, 0, 255],  # 4 ul2
        [85, 85, 0, 255],  # 5 ul3
        [255, 85, 0, 255],  # 6 ul4
        [85, 170, 0, 255],  # 7, ul5
        [255, 170, 0, 255],  # 8, ul6
        [85, 255, 0, 255],  # 9 ul7
        [255, 255, 0, 255],  # 10, ul8
        [0, 0, 255, 255],  # 11 ur1
        [170, 0, 255, 255],  # 12 ur2
        [0, 85, 255, 255],  # 13 ur3
        [170, 85, 255, 255],  # 14 ur4
        [0, 170, 255, 255],  # 15 ur5
        [170, 170, 255, 255],  # 16 ur6
        [0, 255, 255, 255],  # 17 ur7
        [170, 255, 255, 255],  # 18 ur8
        [0, 0, 127, 255],  # 19 bl1
        [170, 0, 127, 255],  # 20 bl2
        [0, 85, 127, 255],  # 21 bl3
        [170, 85, 127, 255],  # 22 bl4
        [0, 170, 127, 255],  # 23 bl5
        [170, 170, 127, 255],  # 24 bl6
        [0, 255, 127, 255],  # 25 bl7
        [170, 255, 127, 255],  # 26 bl8
        [0, 0, 0, 255],  # 27 br1
        [170, 0, 0, 255],  # 28 br2
        [0, 85, 0, 255],  # 29 br3
        [170, 85, 0, 255],  # 30 br4
        [0, 170, 0, 255],  # 31 br5
        [170, 170, 0, 255],  # 32 br6
        [0, 255, 0, 255],  # 33 br7
        [170, 255, 0, 255],  # 34 br8
    ], dtype=np.uint8)
    # 定义三维空间每个轴的显示范围
    extent = [0, class_map.shape[0], 0, class_map.shape[1], 0, class_map.shape[2]]

    # 画3D点图
    p3d = mlab.points3d(x, y, z, label, extent=extent, mode='sphere', scale_factor=1, resolution=4, scale_mode="none")

    # 定义三个轴的注释
    mlab.xlabel("X Label")
    mlab.ylabel("Y Label")
    mlab.zlabel("Z Label")
    # 设置采用自定义的色彩表
    p3d.module_manager.scalar_lut_manager.lut.number_of_colors = config.classes
    p3d.module_manager.scalar_lut_manager.lut.table = color_table
    # 设置点的色彩的模式
    p3d.glyph.color_mode = "color_by_scalar"

    # 定位缺失牙齿并且画出缺失牙齿的位置
    missing_tooth_position, missing_tooth_classes = search_and_display_missing_tooth(class_map, index_to_class_dict)
    print(missing_tooth_position)
    print(missing_tooth_classes)

    # 衡量缺失牙齿在xy平面上的定位效果
    metric_implant_location(missing_tooth_position, missing_tooth_classes, class_map)

    mlab.show()




# 根据缺失牙齿两边的牙齿类别定位缺失牙齿
def locate_missing_tooth(class_map, label1, label2):
    # 获取第一个标签类的坐标信息
    x1_pos, y1_pos, z1_pos = np.where(class_map == label1)
    # 获取第二个标签类的坐标信息
    x2_pos, y2_pos, z2_pos = np.where(class_map == label2)
    # 获取第一个标签坐标在三个轴上的最大值和最小值
    x1_min, x1_max, y1_min, y1_max, z1_min, z1_max = \
        x1_pos.min(), x1_pos.max(), y1_pos.min(), y1_pos.max(), z1_pos.min(), z1_pos.max()
    # 获取第二个标签坐标在三个轴上的最大值和最小值
    x2_min, x2_max, y2_min, y2_max, z2_min, z2_max = \
        x2_pos.min(), x2_pos.max(), y2_pos.min(), y2_pos.max(), z2_pos.min(), z2_pos.max()

    # 分别计算左右两个参考立方体在x轴和y轴上的宽度
    x1_width, y1_width, x2_width, y2_width = x1_max - x1_min, y1_max - y1_min, x2_max - x2_min, y2_max - y2_min
    # 计算缺失牙齿立方体在x轴和y轴上的理论宽度
    x_width, y_width = int((x1_width + x2_width) / 2), int((y1_width + y2_width) / 2)
    # 计算缺失牙齿理论立方体在x轴上的宽度/在y轴上的宽度
    th_ratio = x_width / (y_width + 1e-12)

    # 分别计算缺失牙齿在x轴和y轴上坐标的上下界
    x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound = \
        min(x1_max, x2_max), max(x1_min, x2_min), min(y1_max, y2_max), max(y1_min, y2_min)
    # 分别在x轴和y轴上计算两个参考立方体之间的间隔距离
    x_gap, y_gap = max(0, x_upper_bound - x_lower_bound), max(0, y_upper_bound - y_lower_bound)
    # 计算两个参考立方体在x轴上的间隔距离/在y轴上的间隔距离
    prac_ratio = x_gap / (y_gap + 1e-12)

    # 根据实际间隔距离和缺失牙齿立方体宽度的大小关系，分情况讨论
    if x_gap < x_width and y_gap < y_width:
        # 比较实际间隔长宽比和理论确实牙齿立方体长宽比的大小
        if prac_ratio >= th_ratio:  # 如果x轴间隔比较大，则以x轴间隔作为缺失牙齿立方体在x轴上的宽度
            # 缺失牙齿立方体在x轴上的边界为x轴间隔的左右边界
            x_min, x_max = x_lower_bound + config.missing_tooth_box_margin, x_upper_bound - config.missing_tooth_box_margin
            # 计算缺失牙齿立方体在y轴上的宽度,调整y轴宽度，使得长宽比等于理论立方体的长宽比
            prac_y_width = int(x_gap / th_ratio)
            # 计算缺失牙齿立方体在y轴上的中心点
            y_central_point = int((y1_min + y1_max + y2_min + y2_max) / 4)
            # 根据缺失牙齿立方体在y轴上的中心点和宽度，计算缺失牙齿立方体在y轴上的边界
            y_min, y_max = int(y_central_point - prac_y_width / 2), int(y_central_point + prac_y_width / 2)
        else:
            # 缺失牙齿立方体在y轴上的边界为y轴间隔的左右边界
            y_min, y_max = y_lower_bound + config.missing_tooth_box_margin, y_upper_bound - config.missing_tooth_box_margin
            # 计算缺失牙齿立方体在x轴上的宽度,调整x轴宽度，使得长宽比等于理论立方体的长宽比
            prac_x_width = int(y_gap * th_ratio)
            # 计算缺失牙齿立方体在x轴上的中心点
            x_central_point = int((x1_min + x1_max + x2_min + x2_max) / 4)
            # 根据缺失牙齿立方体在x轴上的中心点和宽度，计算缺失牙齿立方体在x轴上的边界
            x_min, x_max = int(x_central_point - prac_x_width / 2), int(x_central_point + prac_x_width / 2)
    else:  # x轴和y轴中至少有一个轴上的间隔距离大于缺失牙齿理论立方体的宽度
        # 首先x轴和y轴的边界都初始化为间隔的边界
        x_min, x_max, y_min, y_max = x_lower_bound + config.missing_tooth_box_margin, x_upper_bound - config.missing_tooth_box_margin, \
                                     y_lower_bound + config.missing_tooth_box_margin, y_upper_bound - config.missing_tooth_box_margin
        # 如果x轴的间隔小于缺失牙齿理论立方体在x轴上的宽度，则缺失牙齿立方体在x轴上的边界等于缺失牙齿理论立方体在x轴上的边界
        if x_gap < x_width:
            x_min, x_max = int((x1_min + x2_min) / 2), int((x1_max + x2_max) / 2)
        # 如果y轴的间隔小于缺失牙齿理论立方体在y轴上的宽度，则缺失牙齿立方体在y轴上的边界等于缺失牙齿理论立方体在y轴上的边界
        if y_gap < y_width:
            y_min, y_max = int((y1_min + y2_min) / 2), int((y1_max + y2_max) / 2)

    # 计算出缺失牙齿立方体在z轴上的边界
    z_min, z_max = int((z1_min + z2_min) / 2), int((z1_max + z2_max) / 2)

    # 根据这6个面的坐标构造立方体的8个顶点
    #                                   pt1_ _ _ _ _ _ _ _ _pt2
    #                                    /|                 /|
    #                                   / |                / |
    #                               pt3/_ | _ _ _ _ _ _pt4/  |
    #                                 |   |              |   |
    #                                 |   |              |   |
    #                                 |  pt5_ _ _ _ _ _ _|_ _|pt6
    #                                 |  /               |  /
    #                                 | /                | /
    #                     (0,0,0)  pt7|/_ _ _ _ _ _ _ _ _|/pt8
    pt1 = (x_min, y_max, z_max)
    pt2 = (x_max, y_max, z_max)
    pt3 = (x_min, y_min, z_max)
    pt4 = (x_max, y_min, z_max)
    pt5 = (x_min, y_max, z_min)
    pt6 = (x_max, y_max, z_min)
    pt7 = (x_min, y_min, z_min)
    pt8 = (x_max, y_min, z_min)

    return [pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8]



# 根据立方体的八个顶点坐标画出缺失牙齿的定位框
def plot_missing_tooth(vertex_list):
    # 分别获取8个顶点三个轴上的坐标信息
    x1, y1, z1 = vertex_list[0]  # | => pt1
    x2, y2, z2 = vertex_list[1]  # | => pt2
    x3, y3, z3 = vertex_list[2]  # | => pt3
    x4, y4, z4 = vertex_list[3]  # | => pt4
    x5, y5, z5 = vertex_list[4]  # | => pt5
    x6, y6, z6 = vertex_list[5]  # | => pt6
    x7, y7, z7 = vertex_list[6]  # | => pt7
    x8, y8, z8 = vertex_list[7]  # | => pt8
    # 画出立方体目标框
    mlab.mesh([[x1, x2, x6, x6, x6],
               [x3, x4, x8, x6, x2],
               [x7, x8, x8, x6, x2],
               [x7, x7, x7, x5, x1],
               [x7, x3, x3, x1, x1]],
              [[y1, y2, y6, y6, y6],
               [y3, y4, y8, y6, y2],
               [y7, y8, y8, y6, y2],
               [y7, y7, y7, y5, y1],
               [y7, y3, y3, y1, y1]],
              [[z1, z2, z6, z6, z6],
               [z3, z4, z8, z6, z2],
               [z7, z8, z8, z6, z2],
               [z7, z7, z7, z5, z1],
               [z7, z3, z3, z1, z1]],
              color=(1, 0, 0), representation='wireframe', line_width=2.0)





# 搜索并且画出缺失的牙齿
def search_and_display_missing_tooth(class_map, index_to_class_dict):
    # 分别定义上下牙齿从左到右依次的类别索引参考表
    reference_table = [
        [10, 9, 8, 7, 6, 5, 4, 3, 11, 12, 13, 14, 15, 16, 17, 18],
        [26, 25, 24, 23, 22, 21, 20, 19, 27, 28, 29, 30, 31, 32, 33, 34]
    ]
    # 初始化统计各类别是否存在的表
    class_exit_label = [[0]*len(reference_table[0]), [0]*len(reference_table[1])]
    # 根据分割结果，统计各类别的存在状态
    for i in range(len(reference_table)):
        for j in range(len(reference_table[i])):
            if reference_table[i][j] in class_map:
                class_exit_label[i][j] = 1
    print(class_exit_label)
    print(reference_table)

    # 定义保存缺失牙齿8个顶点坐标和牙齿类别的数据结构
    missing_tooth_position = []
    missing_tooth_classes = []
    # 分别遍历上牙和下牙，寻找缺失的牙齿
    for i in range(len(reference_table)):
        # 先确定一排牙齿中左右存在的牙齿边界；对于边上缺牙和天生边上就少牙的情况不予区分
        for j in range(len(class_exit_label[i])):  # 找左边界
            if class_exit_label[i][j] == 1:  # 第一个存在的牙齿为左边界
                left_boundary = j
                break
        # 找右边界
        for j in range(len(class_exit_label[i])-1, -1, -1):
            if class_exit_label[i][j] == 1:
                right_boundary = j
                break
        # 定义需要维护的统计变量
        zero_cnt = 0  # 连续出现0的计数
        pre_one_index = left_boundary  # 上一个1的下标
        # 开始便利搜索缺失牙齿
        for j in range(left_boundary+1, right_boundary+1):
            if class_exit_label[i][j] == 0:
                # 如果是0,计数+1
                zero_cnt += 1
            else:
                # 如果是1,判断之前0的计数
                if zero_cnt > 0:
                    # 之前出现0的计数大于0，说明之前存在缺失牙齿，需要确定缺失牙位置，获得将确实牙齿位置框出来的立方体的八个顶点坐标
                    vertex_list = locate_missing_tooth(class_map, reference_table[i][pre_one_index], reference_table[i][j])
                    # 根据八个顶点画出缺失牙齿的目标框
                    plot_missing_tooth(vertex_list)
                    # 添加当前缺失牙齿位置信息
                    missing_tooth_position.append(vertex_list)
                    # 初始化当前缺失牙齿目标框包含的牙齿类别
                    missing_tooth_box_classes = []
                    # 遍历连续的被识别为一个目标框的所有缺失牙齿
                    for k in range(1, zero_cnt+1):
                        missing_tooth_box_classes.append(index_to_class_dict[reference_table[i][j-k]])
                    # 添加当前缺失牙齿类别信息
                    missing_tooth_classes.append(missing_tooth_box_classes)
                # 0的计数从新开始
                zero_cnt = 0
                # 更新前一个1的索引
                pre_one_index = j

    return missing_tooth_position, missing_tooth_classes



# 根据缺失牙齿位置和种植体判断定位效果
def metric_implant_location(missing_tooth_position, missing_tooth_classes, class_map):
    # 找到所有种植体体素的位置
    x_pos, y_pos, z_pos = np.where(class_map == 2)
    # 确定种植体在xy平面的最小外接矩形
    x_min, x_max, y_min, y_max = x_pos.min(), x_pos.max(), y_pos.min(), y_pos.max()
    # 确定种植牙齿在xy平面最小外接矩形的中心点
    implant_x_center, implant_y_center = (x_min + x_max) / 2, (y_min + y_max) / 2

    # 初始化数据结构
    min_dist = float("inf")  # 中心点最小距离
    min_idx = -1  # 中心点距离最小的缺失牙齿框索引
    min_center_x = -1
    min_center_y = -1
    # 遍历所有缺失牙齿框
    for i in range(len(missing_tooth_position)):
        vertex_list = missing_tooth_position[i]
        # 分别获取3个顶点三个轴上的坐标信息
        x1, y1, z1 = vertex_list[0]  # | => pt1
        x2, y2, z2 = vertex_list[1]  # | => pt2
        x3, y3, z3 = vertex_list[2]  # | => pt3
        # 计算得到当前缺失牙齿目标框在xy平面的中心点
        center_x, center_y = (x1 + x2) / 2, (y1 + y3) / 2
        # 计算与种植体矩形中心点的距离
        dist = math.dist((implant_x_center, implant_y_center), (center_x, center_y))
        # 判断当前距离是否小于之前的距离
        if dist < min_dist:
            min_dist = dist
            min_idx = i
            min_center_x = center_x
            min_center_y = center_y
    print("种植体在xy平面的中心点为：", str((implant_x_center, implant_y_center)))
    print("缺失定位算法对于种植体定位的目标框在xy平面的中心点为：", str((min_center_x, min_center_y)))
    print("种植体处缺失牙齿定位框中心点和种植体中心点的距离为：", min_dist)
    print("种植体处缺失牙齿的类别为：", missing_tooth_classes[min_idx])
























