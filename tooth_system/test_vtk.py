import cv2
import matplotlib.pyplot as plt
import vtk
import numpy as np
from vtkmodules.util import numpy_support
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from PyQt5.QtCore import QSize, Qt, QRect
from PyQt5.QtGui import QIcon, QPixmap, QBitmap, QPainter, QColor, QImage, QBrush, QWindow, QFont
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSizePolicy, QTabWidget, QTabBar, QFileDialog, QPushButton

from my_ui.TabWidget import TabWidget
from my_ui.Viewer3DWidget import Viewer3DWidget
from my_ui.Viewer2DWidget import Viewer2DWidget
from my_ui.ToothAnalysisWidget import ToothAnalysisWidget
from my_ui.ConfigManageWidget import ConfigManageWidget

from lib.preprocess import load_image_or_label

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

index_to_color_dict = {
    0: (0.0, 0.0, 0.0),
    1: (0.3, 0.3, 0.3),
    2: (1.0, 0.8431372549019608, 0.0),
    3: (0.3333333333333333, 0.0, 0.0),
    4: (1.0, 0.0, 0.0),
    5: (0.3333333333333333, 0.3333333333333333, 0.0),
    6: (1.0, 0.3333333333333333, 0.0),
    7: (0.3333333333333333, 0.6666666666666666, 0.0),
    8: (1.0, 0.6666666666666666, 0.0),
    9: (0.3333333333333333, 1.0, 0.0),
    10: (1.0, 1.0, 0.0),
    11: (0.0, 0.0, 1.0),
    12: (0.6666666666666666, 0.0, 1.0),
    13: (0.0, 0.3333333333333333, 1.0),
    14: (0.6666666666666666, 0.3333333333333333, 1.0),
    15: (0.0, 0.6666666666666666, 1.0),
    16: (0.6666666666666666, 0.6666666666666666, 1.0),
    17: (0.0, 1.0, 1.0),
    18: (0.6666666666666666, 1.0, 1.0),
    19: (0.0, 0.0, 0.4980392156862745),
    20: (0.6666666666666666, 0.0, 0.4980392156862745),
    21: (0.0, 0.3333333333333333, 0.4980392156862745),
    22: (0.6666666666666666, 0.3333333333333333, 0.4980392156862745),
    23: (0.0, 0.6666666666666666, 0.4980392156862745),
    24: (0.6666666666666666, 0.6666666666666666, 0.4980392156862745),
    25: (0.0, 1.0, 0.4980392156862745),
    26: (0.6666666666666666, 1.0, 0.4980392156862745),
    27: (0.0, 0.0, 0.0),
    28: (0.6666666666666666, 0.0, 0.0),
    29: (0.0, 0.3333333333333333, 0.0),
    30: (0.6666666666666666, 0.3333333333333333, 0.0),
    31: (0.0, 0.6666666666666666, 0.0),
    32: (0.6666666666666666, 0.6666666666666666, 0.0),
    33: (0.0, 1.0, 0.0),
    34: (0.6666666666666666, 1.0, 0.0)
}


def numpy_to_vtkImageData(np_array):
    # vtkImageData中的数据全部都平铺成了一维数组,所以此处使用ravel()函数进行平铺处理
    depth_arr = numpy_support.numpy_to_vtk(np.ravel(np_array, order="F"), deep=1, array_type=vtk.VTK_SHORT)
    im_data = vtk.vtkImageData()
    # 设置meta信息
    im_data.SetDimensions(np_array.shape)
    im_data.SetSpacing([1, 1, 1])
    im_data.SetOrigin([0, 0, 0])
    # im_data.Set
    # 设置数据信息
    im_data.GetPointData().SetScalars(depth_arr)
    return im_data


# 初始化渲染器、渲染窗口和渲染窗口交互器
render = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(render)
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)


def show_3D_image(image_np):
    # 定义不透明度映射关系
    opacity_dict = {
        i: 1.0
        for i in range(35)
    }
    opacity_dict[0] = 0.0
    # 定义颜色映射关系
    color_dict = index_to_color_dict

    # 转换数据格式
    vtk_image_data = numpy_to_vtkImageData(image_np)
    print(vtk_image_data)

    # 初始化数值和不透明度的映射关系
    opacity_trans_func = vtk.vtkPiecewiseFunction()
    for label_value, op_value in opacity_dict.items():
        opacity_trans_func.AddPoint(label_value, op_value)

    # 初始化数值和颜色的映射关系
    color_trans_func = vtk.vtkColorTransferFunction()
    for label_value, color in color_dict.items():
        color_trans_func.AddRGBPoint(label_value, color[0], color[1], color[2])

    # # 设置梯度不透明度
    # volumeGradientOpacity = vtk.vtkPiecewiseFunction()
    # volumeGradientOpacity.AddPoint(0, 0.0)
    # volumeGradientOpacity.AddPoint(90, 0.5)
    # volumeGradientOpacity.AddPoint(100, 1.0)

    # 设置体积目标的属性
    volume_prop = vtk.vtkVolumeProperty()
    volume_prop.SetColor(color_trans_func)
    volume_prop.SetScalarOpacity(opacity_trans_func)
    # volume_prop.SetGradientOpacity(volumeGradientOpacity)
    volume_prop.ShadeOn()
    volume_prop.SetInterpolationTypeToLinear()

    # volume_prop.SetAmbient(0.4)
    # volume_prop.SetDiffuse(0.6)
    # volume_prop.SetSpecular(0.2)

    # 用图像数据建图
    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetBlendModeToComposite()
    mapper.SetInputData(vtk_image_data)

    # 设置建图的依赖关系
    volume = vtk.vtkVolume()

    # volume.SetPosition(99.5, 99.5, 49.5)
    volume.SetScale(1, 1, 1)
    # volume.SetCoordinateSystemToWorld()
    # volume.SetOrigin(49.75, 49.75, 24.75)

    volume.SetMapper(mapper)
    volume.SetProperty(volume_prop)

    camera = render.GetActiveCamera()
    c = volume.GetCenter()
    camera.SetFocalPoint(c[0], c[1], c[2])
    camera.SetPosition(c[0], c[1] - 500, c[2])
    camera.SetViewUp(0, 0, 1)

    # 创建一个世界坐标系的表示
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(*image_np.shape)
    # 设置坐标轴的标签
    axes.SetXAxisLabelText("X")
    axes.SetYAxisLabelText("Y")
    axes.SetZAxisLabelText("Z")
    # 将坐标系添加到渲染器中
    render.AddActor(axes)

    # 将建好的图进行渲染
    render.AddViewProp(volume)
    render.SetBackground(1, 1, 1)
    print(volume.GetScale())
    print(volume.GetOrigin())
    print(volume.GetCenter())

    # 启动交互器
    # render_window.Render()
    render_window_interactor.Start()


if __name__ == '__main__':
    label_np = load_image_or_label(r"./images/label_12_2.nrrd", [0.5, 0.5, 0.5], type="label", index_to_class_dict=index_to_class_dict)
    print(label_np.shape)
    plt.imshow(label_np[:, :, 50], cmap="gray")
    plt.show()
    show_3D_image(label_np)
