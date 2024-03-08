# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/03 17:02
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import cv2
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


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # 渲染主窗口界面
        self.setupUi()


    def setupUi(self):
        # 设置主窗口
        self.setObjectName("MainWindow")
        self.resize(1800, 1200)
        # 添加主窗口logo
        icon = QIcon()
        icon.addPixmap(QPixmap(r"./designer_ui/icon.png"), QIcon.Normal, QIcon.Off)
        self.setWindowIcon(icon)
        # 设置中心部件
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)
        # 背景垂直布局
        self.background_layout = QVBoxLayout(self.centralwidget)
        self.background_layout.setContentsMargins(0, 0, 0, 0)
        self.background_layout.setSpacing(0)

        # 设置header
        self.header_widget = QWidget()
        self.header_widget.setStyleSheet(
            '''
            background-color: qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5, stop:0 #60a3f7, stop:1 #3b78e6);
            ''')
        self.background_layout.addWidget(self.header_widget, stretch=2)  # 设置header部件
        self.header_layout = QHBoxLayout(self.header_widget)  # header水平布局
        # 添加系统图像
        self.logo_label = QLabel()
        self.logo_label.setFixedSize(50, 50)
        self.logo_label.setScaledContents(True)
        self.logo_label.setStyleSheet(
            '''
            background: transparent;
            ''')
        self.logo_pixmap = QPixmap(r"./designer_ui/icon.png")
        self.logo_label.setPixmap(self.logo_pixmap)
        self.header_layout.addWidget(self.logo_label, stretch=1)
        # 添加系统名称
        self.system_name_label = QLabel("牙齿CBCT图像分析系统")
        self.system_name_label.setStyleSheet(
            '''
            font-family: 微软雅黑;
            color:white;
            font-size:28px;
            letter-spacing:10px;
            margin-left:20px;
            background: transparent;
            ''')
        self.header_layout.addWidget(self.system_name_label, stretch=7)
        # 添加打开图像路径和按钮
        self.opening_image_path_label = QLabel()
        self.opening_image_path_label.setMinimumSize(300, 0)
        self.opening_image_path_label.setWordWrap(True)
        self.opening_image_path_label.setStyleSheet(
            '''
            font-family: 微软雅黑;
            color:white;
            font-size:16px;
            font-weight:400;
            background: transparent;
            ''')
        self.header_layout.addWidget(self.opening_image_path_label, stretch=10)
        self.select_image_btn = QPushButton("打开图像")
        self.select_image_btn.setStyleSheet(
            '''
            QPushButton {
                height:30px;
                background-color: qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5, stop:0 #DBDBDB, stop:1 #3B3B3B);
                border-radius: 5px;
                color: white;
                font-size:18px;
                font-weight: bold;
                padding: 5px;
            }
            
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5, stop:0 #3B3B3B, stop:1 #DBDBDB);
            }
            ''')
        self.select_image_btn.clicked.connect(self.select_image)  # 添加点击信号
        self.header_layout.addWidget(self.select_image_btn, stretch=1)
        self.header_layout.addStretch(1)  # 添加一段留白
        # 添加用户头像
        self.avater_label = QLabel()
        self.avater_label.setFixedSize(50, 50)
        self.avater_label.setScaledContents(True)
        self.avater_label.setStyleSheet(
            '''
            background: transparent;
            border-radius: 25px;
            ''')
        self.avater_pixmap = self.mask_image_circle(r"./designer_ui/avater.jpg", imgtype="jpg", size=50)
        self.avater_label.setPixmap(self.avater_pixmap)
        self.header_layout.addWidget(self.avater_label, stretch=1)
        # 添加用户名
        self.username_label = QLabel("yykzjh")
        self.username_label.setFont(QFont("微软雅黑"))
        self.username_label.setStyleSheet(
            '''
            color:white;
            font-size:24px;
            letter-spacing:5px;
            margin-left:10px;
            background: transparent;
            ''')
        self.header_layout.addWidget(self.username_label, stretch=5)

        # 设置body
        self.body_tab_widget = TabWidget()
        self.body_tab_widget.setTabPosition(QTabWidget.West)
        self.body_tab_widget.setStyleSheet(
            '''
            QTabWidget::pane {
                border-left: 1px solid #EAEAEA;
                position:absolute;
                left:-0.1px;
            }
            QTabWidget QTabBar {
                font-size:24px;
                font-family:微软雅黑;
                font-weight:400;
            }
            QTabWidget QTabBar::tab {
                width:250px;
                height:60px;
                background:#FFFFFF;
                border:1px solid gray;
                border-right-color:#FFFFFF;
                border-top-left-radius:20px;
                border-bottom-left-radius:20px;
                padding:2px;
            }
            QTabBar::tab:selected {
                color:#333333;
                border-color:gray;
                border-right: 2px solid;
                border-right-color:#4BA4F2;
            }
            QTabBar::tab:hover {
                background-color:#EEEEEE;
            }
            QTabBar::tab:!selected {
                color:#B2B2B2;
                border-color:gray;
                border-right-color:#FFFFFF;
            }
            ''')
        self.background_layout.addWidget(self.body_tab_widget, stretch=40)
        # 添加牙齿表面轮廓分割tab
        self.tab1 = Viewer3DWidget()
        surface_label_np = load_image_or_label(r"./images/surface_label_12_2.nii.gz", [0.5, 0.5, 0.5], type="surface_label", index_to_class_dict=index_to_class_dict)
        self.show_3D_image(surface_label_np, self.tab1)
        self.body_tab_widget.addTab(self.tab1, "牙齿表面轮廓分割")
        # 添加牙齿几何中心检测tab
        self.tab2 = Viewer3DWidget()
        centroid_label_np = load_image_or_label(r"./images/centroid_label_12_2.txt", [0.5, 0.5, 0.5], type="centroid_label", index_to_class_dict=index_to_class_dict)
        c, h, w, d = centroid_label_np.shape
        centroid_label_np = np.concatenate([np.full((1, h, w, d), 0.6), centroid_label_np], axis=0)
        centroid_label_np = np.argmax(centroid_label_np, axis=0)
        self.show_3D_image(centroid_label_np, self.tab2)
        self.body_tab_widget.addTab(self.tab2, "牙齿几何中心检测")
        # 添加牙齿体积分割tab
        self.tab3 = Viewer3DWidget()
        label_np = load_image_or_label(r"./images/label_12_2.nrrd", [0.5, 0.5, 0.5], type="label", index_to_class_dict=index_to_class_dict)
        self.show_3D_image(label_np, self.tab3)
        self.body_tab_widget.addTab(self.tab3, "牙齿体积分割")
        # 牙齿图像分析tab
        self.tab4 = ToothAnalysisWidget()
        self.body_tab_widget.addTab(self.tab4, "牙齿图像分析")
        # 添加生成冠状面全景图tab
        self.tab5 = Viewer2DWidget()
        self.tab5.viewer_widget.setMaximumHeight(700)
        self.panoramic_image = cv2.imread(r"./images/panoramic_image.jpg")
        self.tab5.img = self.panoramic_image
        h, w, c = self.tab5.img.shape
        self.tab5.image = QImage(self.tab5.img, w, h, c*w, QImage.Format_RGB888)  # 如果没有depth*width，图像可能会扭曲
        self.tab5.pixmap = QPixmap(self.tab5.image)  # 创建相应的QPixmap对象
        self.tab5.viewer_widget.setPixmap(self.tab5.pixmap)  # 显示图像
        self.body_tab_widget.addTab(self.tab5, "生成冠状面全景图")
        # 添加牙齿图像细节增强tab
        self.tab6 = Viewer2DWidget()
        self.enhance_image = cv2.imread(r"./images/10_enhance.jpg")
        self.tab6.img = self.enhance_image
        h, w, c = self.tab6.img.shape
        self.tab6.image = QImage(self.tab6.img, w, h, c * w, QImage.Format_RGB888)  # 如果没有depth*width，图像可能会扭曲
        self.tab6.pixmap = QPixmap(self.tab6.image)  # 创建相应的QPixmap对象
        self.tab6.viewer_widget.setPixmap(self.tab6.pixmap)  # 显示图像
        self.body_tab_widget.addTab(self.tab6, "牙齿图像细节增强")
        # 添加配置管理tab
        self.tab7 = ConfigManageWidget()
        self.body_tab_widget.addTab(self.tab7, "配置管理")

        # 设置footer
        self.footer_widget = QWidget()
        self.footer_widget.setStyleSheet(
            '''
            background-color: qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5, stop:0 #F5C30B, stop:1 #FC8004);
            ''')
        self.background_layout.addWidget(self.footer_widget, stretch=1)  # 设置footer部件

    def numpy_to_vtkImageData(self, np_array):
        # vtkImageData中的数据全部都平铺成了一维数组,所以此处使用ravel()函数进行平铺处理
        depth_arr = numpy_support.numpy_to_vtk(np.ravel(np_array, order="F"), deep=1, array_type=vtk.VTK_DOUBLE)
        im_data = vtk.vtkImageData()
        # 设置meta信息
        im_data.SetDimensions(np_array.shape)
        im_data.SetSpacing([1, 1, 1])
        im_data.SetOrigin([0, 0, 0])
        # 设置数据信息
        im_data.GetPointData().SetScalars(depth_arr)
        return im_data


    def show_3D_image(self, image_np, tab):
        # 定义不透明度映射关系
        opacity_dict = {
            i: 1.0
            for i in range(35)
        }
        opacity_dict[0] = 0.0
        # 定义颜色映射关系
        color_dict = index_to_color_dict

        # 转换数据格式
        vtk_image_data = self.numpy_to_vtkImageData(image_np)

        # 初始化渲染器、渲染窗口和渲染窗口交互器
        render = vtk.vtkRenderer()  # 初始化渲染器
        tab.viewer_widget.GetRenderWindow().AddRenderer(render)  # 将渲染窗口添加到渲染窗口交互器中

        # 设置体积目标的属性
        volume_prop = vtk.vtkVolumeProperty()
        volume_prop.ShadeOn()
        volume_prop.SetInterpolationTypeToLinear()
        volume_prop.SetAmbient(0.4)
        volume_prop.SetDiffuse(0.6)
        volume_prop.SetSpecular(0.2)

        # 初始化数值和不透明度的映射关系
        opacity_trans_func = vtk.vtkPiecewiseFunction()
        for label_value, op_value in opacity_dict.items():
            opacity_trans_func.AddPoint(label_value, op_value)
        volume_prop.SetScalarOpacity(opacity_trans_func)

        # 初始化数值和颜色的映射关系
        color_trans_func = vtk.vtkColorTransferFunction()
        for label_value, color in color_dict.items():
            color_trans_func.AddRGBPoint(label_value, color[0], color[1], color[2])
        volume_prop.SetColor(color_trans_func)

        volumeCompositeOpacity = vtk.vtkPiecewiseFunction()
        volumeCompositeOpacity.AddPoint(50, 0.15)
        volumeCompositeOpacity.AddPoint(120, 0.6)
        volumeCompositeOpacity.AddPoint(200, 1.0)
        volume_prop.SetGradientOpacity(volumeCompositeOpacity)

        # 设置梯度不透明度
        volumeGradientOpacity = vtk.vtkPiecewiseFunction()
        volumeGradientOpacity.AddPoint(1, 0.15)
        volumeGradientOpacity.AddPoint(70, 0.6)
        volumeGradientOpacity.AddPoint(130, 1.0)
        volume_prop.SetGradientOpacity(volumeGradientOpacity)

        # 用图像数据建图
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        mapper.SetBlendModeToComposite()
        mapper.SetInputData(vtk_image_data)

        # 设置建图的依赖关系
        volume = vtk.vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(volume_prop)

        # 设置摄像头位置
        camera = render.GetActiveCamera()
        c = volume.GetCenter()
        camera.SetFocalPoint(c[0], c[1], c[2])
        camera.SetPosition(c[0]-200, c[1] - 500, c[2]+200)
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

        tab.viewer_widget.Initialize()



    def select_image(self):
        fname, _ = QFileDialog.getOpenFileName(self.header_widget, '选择图像', r'./images', '*')
        if fname != "":
            self.opening_image_path_label.setText(fname)


    def mask_image_circle(self, imgpath, imgtype="jpg", size=50):
        imgdata = open(imgpath, 'rb').read()

        # Load image
        image = QImage.fromData(imgdata, imgtype)

        # convert image to 32-bit ARGB (adds an alpha
        # channel ie transparency factor):
        image.convertToFormat(QImage.Format_ARGB32)

        # Crop image to a square:
        imgsize = min(image.width(), image.height())
        rect = QRect(
            (image.width() - imgsize) / 2,
            (image.height() - imgsize) / 2,
            imgsize,
            imgsize,
        )

        image = image.copy(rect)

        # Create the output image with the same dimensions
        # and an alpha channel and make it completely transparent:
        out_img = QImage(imgsize, imgsize, QImage.Format_ARGB32)
        out_img.fill(Qt.transparent)

        # Create a texture brush and paint a circle
        # with the original image onto the output image:
        brush = QBrush(image)

        # Paint the output image
        painter = QPainter(out_img)
        painter.setBrush(brush)

        # Don't draw an outline
        painter.setPen(Qt.NoPen)

        # drawing circle
        painter.drawEllipse(0, 0, imgsize, imgsize)

        # closing painter event
        painter.end()

        # Convert the image to a pixmap and rescale it.
        pr = QWindow().devicePixelRatio()
        pm = QPixmap.fromImage(out_img)
        pm.setDevicePixelRatio(pr)
        size *= pr
        pm = pm.scaled(size, size, Qt.KeepAspectRatio,
                       Qt.SmoothTransformation)

        # return back the pixmap data
        return pm

    def closeEvent(self, QCloseEvent):
        super(MainWindow, self).closeEvent(QCloseEvent)
        self.tab1.close()
        self.tab2.close()
        self.tab3.close()
        self.tab4.close()
        self.tab5.close()
        self.tab6.close()
        self.tab7.close()



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
