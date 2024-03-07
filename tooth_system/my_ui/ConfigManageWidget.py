# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/04 20:39
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from PyQt5.QtCore import QSize, Qt, QRect
from PyQt5.QtGui import QIcon, QPixmap, QBitmap, QPainter, QColor, QImage, QBrush, QWindow, QFont
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSizePolicy, QTabWidget, QTabBar, QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox


class ConfigManageWidget(QWidget):
    def __init__(self):
        super(ConfigManageWidget, self).__init__()
        # 渲染部件界面
        self.setupUi()

    def setupUi(self):
        self.setStyleSheet(
            '''
            QLabel, QComboBox, QSpinBox, QDoubleSpinBox{
                font-size:24px;
                font-family:Microsoft YaHei;
            }
            ''')
        # 设置组件界面布局
        self.viewer_layout = QVBoxLayout(self)
        self.viewer_layout.setContentsMargins(50, 20, 50, 20)

        # 任务目标：二分类、多类别
        self.task_target_widget = QWidget()
        self.task_target_layout = QHBoxLayout(self.task_target_widget)
        self.task_target_layout.addWidget(QLabel("任务目标："))
        self.task_target_combobox = QComboBox()
        self.task_target_combobox.addItems(["二分类", "多类别"])
        self.task_target_layout.addWidget(self.task_target_combobox)
        self.task_target_layout.addStretch(100)
        self.viewer_layout.addWidget(self.task_target_widget, stretch=1)
        # 算法架构：单阶段、两阶段
        self.algorithm_architecure_widget = QWidget()
        self.algorithm_architecure_layout = QHBoxLayout(self.algorithm_architecure_widget)
        self.algorithm_architecure_layout.addWidget(QLabel("算法架构："))
        self.algorithm_architecure_combobox = QComboBox()
        self.algorithm_architecure_combobox.addItems(["单阶段", "两阶段"])
        self.algorithm_architecure_layout.addWidget(self.algorithm_architecure_combobox)
        self.algorithm_architecure_layout.addStretch(100)
        self.viewer_layout.addWidget(self.algorithm_architecure_widget, stretch=1)
        # 灰度值clip上下界
        self.clip_widget = QWidget()
        self.clip_layout = QHBoxLayout(self.clip_widget)
        self.clip_layout.addWidget(QLabel("灰度值Clip下界(clip_lower_bound)："))
        self.clip_lower_bound_spinbox = QSpinBox()
        self.clip_lower_bound_spinbox.setMaximum(18000)
        self.clip_lower_bound_spinbox.setMinimum(-4000)
        self.clip_layout.addWidget(self.clip_lower_bound_spinbox)
        self.clip_layout.addStretch(20)
        self.clip_layout.addWidget(QLabel("灰度值Clip上界(clip_upper_bound)："))
        self.clip_upper_bound_spinbox = QSpinBox()
        self.clip_upper_bound_spinbox.setMaximum(18000)
        self.clip_upper_bound_spinbox.setMinimum(-4000)
        self.clip_layout.addWidget(self.clip_upper_bound_spinbox)
        self.clip_layout.addStretch(100)
        self.viewer_layout.addWidget(self.clip_widget, stretch=1)
        # Normalize的均值和标准差
        self.normalize_widget = QWidget()
        self.normalize_layout = QHBoxLayout(self.normalize_widget)
        self.normalize_layout.addWidget(QLabel("Normalize的均值(mean)："))
        self.normalize_mean_doublespinbox = QDoubleSpinBox()
        self.normalize_mean_doublespinbox.setMaximum(1)
        self.normalize_mean_doublespinbox.setMinimum(0)
        self.normalize_mean_doublespinbox.setDecimals(6)
        self.normalize_mean_doublespinbox.setSingleStep(0.01)
        self.normalize_layout.addWidget(self.normalize_mean_doublespinbox)
        self.normalize_layout.addStretch(20)
        self.normalize_layout.addWidget(QLabel("Normalize的标准差(std)："))
        self.normalize_std_doublespinbox = QDoubleSpinBox()
        self.normalize_std_doublespinbox.setMaximum(1)
        self.normalize_std_doublespinbox.setMinimum(0)
        self.normalize_std_doublespinbox.setDecimals(6)
        self.normalize_std_doublespinbox.setSingleStep(0.01)
        self.normalize_layout.addWidget(self.normalize_std_doublespinbox)
        self.normalize_layout.addStretch(100)
        self.viewer_layout.addWidget(self.normalize_widget, stretch=1)
        # 分割模型：["PMFSNet", "UNet3D", "DenseVNet", "AttentionUNet3D", "DenseVoxelNet", "MultiResUNet3D", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet"]
        self.model_widget = QWidget()
        self.model_layout = QHBoxLayout(self.model_widget)
        self.model_layout.addWidget(QLabel("分割模型："))
        self.model_combobox = QComboBox()
        self.model_combobox.addItems(["PMFSNet", "UNet3D", "DenseVNet", "AttentionUNet3D", "DenseVoxelNet", "MultiResUNet3D", "UNETR", "SwinUNETR", "TransBTS", "nnFormer", "3DUXNet"])
        self.model_layout.addWidget(self.model_combobox)
        self.model_layout.addStretch(100)
        self.viewer_layout.addWidget(self.model_widget, stretch=1)
        # PMFSNet模型缩放版本
        self.scaling_version_widget = QWidget()
        self.scaling_version_layout = QHBoxLayout(self.scaling_version_widget)
        self.scaling_version_layout.addWidget(QLabel("PMFSNet模型缩放版本："))
        self.scaling_version_combobox = QComboBox()
        self.scaling_version_combobox.addItems(["TINY", "SMALL", "BASIC"])
        self.scaling_version_layout.addWidget(self.scaling_version_combobox)
        self.scaling_version_layout.addStretch(100)
        self.viewer_layout.addWidget(self.scaling_version_widget, stretch=1)
        # 是否采用PMFS Block
        self.use_PMFS_Block_widget = QWidget()
        self.use_PMFS_Block_layout = QHBoxLayout(self.use_PMFS_Block_widget)
        self.use_PMFS_Block_layout.addWidget(QLabel("是否采用PMFS Block："))
        self.use_PMFS_Block_combobox = QComboBox()
        self.use_PMFS_Block_combobox.addItems(["是", "否"])
        self.use_PMFS_Block_layout.addWidget(self.use_PMFS_Block_combobox)
        self.use_PMFS_Block_layout.addStretch(100)
        self.viewer_layout.addWidget(self.use_PMFS_Block_widget, stretch=1)
        # 滑动窗口步长：4~64
        self.slide_step_widget = QWidget()
        self.slide_step_layout = QHBoxLayout(self.slide_step_widget)
        self.slide_step_layout.addWidget(QLabel("滑动窗口步长："))
        self.slide_step_spinbox = QSpinBox()
        self.slide_step_spinbox.setMaximum(64)
        self.slide_step_spinbox.setMinimum(4)
        self.slide_step_layout.addWidget(self.slide_step_spinbox)
        self.slide_step_layout.addStretch(100)
        self.viewer_layout.addWidget(self.slide_step_widget, stretch=1)
        # 提交和重置按钮
        self.btn_widget = QWidget()
        self.btn_layout = QHBoxLayout(self.btn_widget)
        self.btn_layout.addStretch(2)
        # 添加提交按钮
        self.submit_btn = QPushButton("提交")
        self.submit_btn.setStyleSheet(
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
        self.btn_layout.addWidget(self.submit_btn, stretch=1)
        self.btn_layout.addStretch(2)
        # 添加重置按钮
        self.reset_btn = QPushButton("重置")
        self.reset_btn.setStyleSheet(
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
        self.btn_layout.addWidget(self.reset_btn, stretch=1)
        self.btn_layout.addStretch(5)
        self.viewer_layout.addWidget(self.btn_widget, stretch=1)

        # 部件主界面底部留白
        self.viewer_layout.addStretch(2)


