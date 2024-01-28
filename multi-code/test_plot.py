# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/9/27 0:54
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import numpy as np
import cc3d
import matplotlib.pyplot as plt
import matplotlib as mpl

from mayavi import mlab



def numpy_plot_static_3D():
    class_map = np.array(
        [
            [[1, 1, 0],
             [1, 0, 1],
             [1, 1, 1]],
            [[0, 0, 1],
             [0, 1, 1],
             [1, 1, 1]],
            [[1, 1, 1],
             [0, 0, 0],
             [0, 1, 0]]
        ]
    )

    x = []
    y = []
    z = []
    label = []
    class_num = 2
    for i in range(class_num):
        pos_x, pos_y, pos_z = np.nonzero(class_map == i)
        x.extend(list(pos_x))
        y.extend(list(pos_y))
        z.extend(list(pos_z))
        label.extend([i] * len(pos_x))

    # tooth_colors = mpl.colors.LinearSegmentedColormap.from_list(
    #     '牙齿种类颜色',
    #     ['#1f77b4', '#ff7f0e'],
    #     N=256
    # )

    tooth_colormap = mpl.colors.ListedColormap(['#1f77b4', '#ff7f0e'])
    plt.cm.register_cmap(name='tooth', cmap=tooth_colormap)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=label, marker='o', cmap=plt.cm.get_cmap('tooth'), alpha=1, s=100)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.show()





def mayavi_plot_dynamic_3D():
    # x = [[-1, 1, 1, -1, -1], [-1, 1, 1, -1, -1]]
    # y = [[-1, -1, -1, -1, -1], [1, 1, 1, 1, 1]]
    # z = [[1, 1, -1, -1, 1], [1, 1, -1, -1, 1]]
    #
    # s = mlab.mesh(x, y, z, representation='wireframe', line_width=1.0, extent=[-1, 1, -1, 1, -1, 1])
    # mlab.xlabel("X Label")
    # mlab.ylabel("Y Label")
    # mlab.zlabel("Z Label")
    # mlab.show()


    x1, y1, z1 = (0, 1, 1)  # | => pt1
    x2, y2, z2 = (1, 1, 1)  # | => pt2
    x3, y3, z3 = (0, 0, 1)  # | => pt3
    x4, y4, z4 = (1, 0, 1)  # | => pt4
    x5, y5, z5 = (0, 1, 0)  # | => pt5
    x6, y6, z6 = (1, 1, 0)  # | => pt6
    x7, y7, z7 = (0, 0, 0)  # | => pt7
    x8, y8, z8 = (1, 0, 0)  # | => pt8

    # mlab.mesh([[x1, x2], [x3, x4]],  # | => x coordinate
    #           [[y1, y2], [y3, y4]],  # | => y coordinate
    #           [[z1, z2], [z3, z4]],  # | => z coordinate
    #           color=(0, 0, 0), representation='wireframe', line_width=1.0)  # black
    # mlab.mesh([[x5, x6], [x7, x8]],
    #                  [[y5, y6], [y7, y8]],
    #                  [[z5, z6], [z7, z8]],
    #                  color=(1, 0, 0), representation='wireframe', line_width=1.0)  # red
    #
    # mlab.mesh([[x1, x3], [x5, x7]],
    #                  [[y1, y3], [y5, y7]],
    #                  [[z1, z3], [z5, z7]],
    #                  color=(0, 0, 1), representation='wireframe', line_width=1.0)  # blue
    #
    # mlab.mesh([[x1, x2], [x5, x6]],
    #                  [[y1, y2], [y5, y6]],
    #                  [[z1, z2], [z5, z6]],
    #                  color=(1, 1, 0), representation='wireframe', line_width=1.0)  # yellow
    #
    # mlab.mesh([[x2, x4], [x6, x8]],
    #                  [[y2, y4], [y6, y8]],
    #                  [[z2, z4], [z6, z8]],
    #                  color=(1, 1, 1), representation='wireframe', line_width=1.0)  # white
    #
    # mlab.mesh([[x3, x4], [x7, x8]],
    #                  [[y3, y4], [y7, y8]],
    #                  [[z3, z4], [z7, z8]],
    #                  color=(1, 0, 1), representation='wireframe', line_width=1.0)  # pink

    # mlab.mesh([[x1, x2, x6], [x3, x4, x8], [x7, x8, x8]],
    #                  [[y1, y2, y6], [y3, y4, y8], [y7, y8, y8]],
    #                  [[z1, z2, z6], [z3, z4, z8], [z7, z8, z8]],
    #                  color=(1, 0, 0), representation='wireframe', line_width=1.0)  # red
    # mlab.mesh([[x8, x6, x2], [x7, x5, x1], [x3, x1, x1]],
    #           [[y8, y6, y2], [y7, y5, y1], [y3, y1, y1]],
    #           [[z8, z6, z2], [z7, z5, z1], [z3, z1, z1]],
    #           color=(0, 0, 1), representation='wireframe', line_width=1.0)  # blue

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
              color=(1, 0, 0), representation='wireframe', line_width=5.0)

    mlab.xlabel("X Label")
    mlab.ylabel("Y Label")
    mlab.zlabel("Z Label")

    mlab.show()


    # def test_points3d():
    #     t = np.linspace(0, 4 * np.pi, 20)
    #     x = np.sin(2 * t)
    #     y = np.cos(t)
    #     z = np.cos(2 * t)
    #     s = 2 + np.sin(t)
    #     return mlab.points3d(x, y, z, s, colormap="Reds", scale_factor=.25)  # s (x,y,z)处标量的值 copper
    #
    # test_points3d()
    # mlab.show()



def test_CCL():
    segment_map = np.array([
        [[2, 2, 2],
         [2, 0, 0],
         [2, 2, 2]],
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],
        [[0, 0, 0],
         [0, 2, 0],
         [0, 0, 0]]
    ])
    print(segment_map[segment_map == 1])

    # 定义连通性强度
    connectivity = 26
    # 进行连通区域分析标记，返回标记三维数组
    labels_out, CCN = cc3d.connected_components(segment_map, return_N=True, connectivity=connectivity)
    print(labels_out)
    # 进一步获取连通区域的统计信息
    """
    {
      voxel_counts: np.ndarray[uint64_t] (index is label) (N+1)

      # Structure is xmin,xmax,ymin,ymax,zmin,zmax by label
      bounding_boxes: np.ndarray[uint64_t] (N+1 x 6)

      # Structure is x,y,z
      centroids: np.ndarray[float64] (N+1,3)
    }
    """
    statistics_data = cc3d.statistics(labels_out)
    print(statistics_data)
    # 得到每个连通区域的大小，索引表示连通区域的编号，元素值为该连通区域大小
    voxel_counts = statistics_data["voxel_counts"]



def test():
    a = 1
    return 1 if a != 0 else 0, 100



if __name__ == '__main__':

    # numpy_plot_static_3D()

    # mayavi_plot_dynamic_3D()

    # color_table = [
    #     [255, 255, 255, 0], # 0 background
    #     [255, 255, 255, 30], # 1 gum
    #     [255, 215, 0, 255], # 2 implant
    #     [85, 0, 0, 255], # 3 ul1
    #     [255, 0, 0, 255], # 4 ul2
    #     [85, 85, 0, 255], # 5 ul3
    #     [255, 85, 0, 255], # 6 ul4
    #     [85, 170, 0, 255], # 7, ul5
    #     [255, 170, 0, 255], # 8, ul6
    #     [85, 255, 0, 255], # 9 ul7
    #     [255, 255, 0, 255], # 10, ul8
    #     [0 ,0, 255, 255], # 11 ur1
    #     [170, 0, 255, 255], # 12 ur2
    #     [0, 85, 255, 255], # 13 ur3
    #     [170, 85, 255, 255], # 14 ur4
    #     [0, 170, 255, 255], # 15 ur5
    #     [170, 170, 255, 255], # 16 ur6
    #     [0, 255, 255, 255], # 17 ur7
    #     [170, 255, 255, 255], # 18 ur8
    #     [0, 0, 127, 255], # 19 bl1
    #     [170, 0, 127, 255], # 20 bl2
    #     [0, 85, 127, 255], # 21 bl3
    #     [170, 85, 127, 255], # 22 bl4
    #     [0, 170, 127, 255], # 23 bl5
    #     [170, 170, 127, 255], # 24 bl6
    #     [0, 255, 127, 255], # 25 bl7
    #     [170, 255, 127, 255], # 26 bl8
    #     [0, 0, 0, 255], # 27 br1
    #     [170, 0, 0, 255], # 28 br2
    #     [0, 85, 0, 255], # 29 br3
    #     [170, 85, 0, 255], # 30 br4
    #     [0, 170, 0, 255], # 31 br5
    #     [170, 170, 0, 255], # 32 br6
    #     [0, 255, 0, 255], # 33 br7
    #     [170, 255, 0, 255], # 34 br8
    # ]

    # test_CCL()


    print(test())








