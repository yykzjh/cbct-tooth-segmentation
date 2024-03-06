# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/04 13:01
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import numpy as np

from PyQt5.QtCore import QSize, Qt, QRect
from PyQt5.QtGui import QIcon, QPixmap, QBitmap, QPainter, QColor, QImage, QBrush, QWindow, QFont
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSizePolicy, QTabWidget, QTabBar, QPushButton


class Viewer2DWidget(QWidget):
    def __init__(self):
        super(Viewer2DWidget, self).__init__()
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

        # 添加图像展示窗口
        self.viewer_widget = QLabel()
        self.viewer_widget.setScaledContents(True)
        self.left_layout.addWidget(self.viewer_widget)
        # 显示图像
        self.img = np.random.random((500, 500, 3))
        h, w, c = self.img.shape
        self.image = QImage(self.img, w, h, c*w, QImage.Format_RGB888)  # 如果没有depth*width，图像可能会扭曲
        self.pixmap = QPixmap(self.image)  # 创建相应的QPixmap对象
        self.viewer_widget.setPixmap(self.pixmap)  # 显示图像

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