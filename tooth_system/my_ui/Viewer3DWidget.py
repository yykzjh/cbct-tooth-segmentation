# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/04 13:01
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from PyQt5.QtCore import QSize, Qt, QRect
from PyQt5.QtGui import QIcon, QPixmap, QBitmap, QPainter, QColor, QImage, QBrush, QWindow, QFont
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSizePolicy, QTabWidget, QTabBar, QPushButton

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class Viewer3DWidget(QWidget):
    def __init__(self):
        super(Viewer3DWidget, self).__init__()
        # 渲染部件界面
        self.setupUi()

    def setupUi(self):
        # 设置组件界面布局
        self.viewer_layout = QHBoxLayout(self)

        # 创建左边的图像展示界面
        self.left_widget = QWidget()
        # 设置左边的图像展示界面布局
        self.left_layout = QVBoxLayout()
        self.left_layout.setContentsMargins(10, 10, 10, 10)
        self.left_layout.setSpacing(20)
        self.left_widget.setLayout(self.left_layout)
        # 将左边的图像展示界面添加到部件主界面
        self.viewer_layout.addWidget(self.left_widget, stretch=5)

        # # 添加图像展示窗口
        self.viewer_widget = QVTKRenderWindowInteractor()
        self.left_layout.addWidget(self.viewer_widget)
        # # 创建数据源
        # self.source = vtk.vtkSphereSource()
        # self.source.SetCenter(0, 0, 0)
        # self.source.SetRadius(3.0)
        # # 创建映射器
        # self.mapper = vtk.vtkPolyDataMapper()
        # self.mapper.SetInputConnection(self.source.GetOutputPort())
        # # 创建actor
        # self.actor = vtk.vtkActor()
        # self.actor.SetMapper(self.mapper)
        # # 创建渲染器
        # self.render = vtk.vtkRenderer()
        # self.viewer_widget.GetRenderWindow().AddRenderer(self.render)
        # self.render.AddActor(self.actor)
        # # 获取交互器并初始化
        # self.interactor = self.viewer_widget.GetRenderWindow().GetInteractor()
        # self.interactor.Initialize()
        # # 添加世界坐标系
        # self.axesActor = vtk.vtkAxesActor()
        # self.axes_widget = vtk.vtkOrientationMarkerWidget()
        # self.axes_widget.SetOrientationMarker(self.axesActor)
        # self.axes_widget.SetInteractor(self.interactor)
        # self.axes_widget.EnabledOn()
        # self.axes_widget.InteractiveOff()  # 坐标系是否可移动

        # 创建右边的图像展示界面
        self.right_widget = QWidget()
        # 设置右边的图像展示界面布局
        self.right_layout = QVBoxLayout()
        self.right_layout.setContentsMargins(50, 10, 50, 10)
        self.right_layout.setSpacing(20)
        self.right_widget.setLayout(self.right_layout)
        # 将右边的图像展示界面添加到部件主界面
        self.viewer_layout.addWidget(self.right_widget, stretch=1)
        # 添加执行按钮
        self.execute_btn = QPushButton("执行")
        self.execute_btn.setStyleSheet(
            '''
            QPushButton {
                height:40px;
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3b78e6, stop:1 #0f5fd7);
                border-radius: 10px;
                color: white;
                font-size:24px;
                font-weight: bold;
                padding: 5px 15px;
            }
            
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #60a3f7, stop:1 #3b78e6);
            }
            
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f5fd7, stop:1 #3b78e6);
            }
            ''')
        self.right_layout.addWidget(self.execute_btn, stretch=1)
        # 添加保存按钮
        self.save_btn = QPushButton("保存")
        self.save_btn.setStyleSheet(
            '''
            QPushButton {
                height:40px;
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3b78e6, stop:1 #0f5fd7);
                border-radius: 10px;
                color: white;
                font-size:24px;
                font-weight: bold;
                padding: 5px 15px;
            }
            
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #60a3f7, stop:1 #3b78e6);
            }
            
            QPushButton:pressed {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f5fd7, stop:1 #3b78e6);
            }
            ''')
        self.right_layout.addWidget(self.save_btn, stretch=1)

    def closeEvent(self, QCloseEvent):
        super(Viewer3DWidget, self).closeEvent(QCloseEvent)
        if self.viewer_widget is not None:
            self.viewer_widget.Finalize()



