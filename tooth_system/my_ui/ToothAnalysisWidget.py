# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/04 15:08
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from PyQt5.QtCore import QSize, Qt, QRect
from PyQt5.QtGui import QIcon, QPixmap, QBitmap, QPainter, QColor, QImage, QBrush, QWindow, QFont
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSizePolicy, QTabWidget, QTabBar, QPushButton, QLineEdit



class ToothAnalysisWidget(QWidget):
    def __init__(self):
        super(ToothAnalysisWidget, self).__init__()
        # 渲染部件界面
        self.setupUi()

    def setupUi(self):
        self.setStyleSheet(
            '''
            QLabel {
                font-size:24px;
                font-family:Microsoft YaHei;
            }
            ''')
        # 设置组件界面布局
        self.viewer_layout = QVBoxLayout(self)
        self.viewer_layout.setContentsMargins(50, 20, 50, 20)

        # 缺失牙齿类型
        self.missing_tooth_widget = QWidget()
        self.missing_tooth_layout = QHBoxLayout(self.missing_tooth_widget)
        self.missing_tooth_layout.addWidget(QLabel("缺失牙齿类型："))
        self.missing_tooth_value = QLabel("ul8, ur8, br5")
        self.missing_tooth_value.setStyleSheet("font-weight:bold;")
        self.missing_tooth_layout.addWidget(self.missing_tooth_value)
        self.missing_tooth_layout.addStretch(100)
        self.viewer_layout.addWidget(self.missing_tooth_widget, stretch=1)
        # 术前距牙槽嵴顶端2mm处的牙槽骨宽度(A值)
        self.A_widget = QWidget()
        self.A_layout = QHBoxLayout(self.A_widget)
        self.A_layout.addWidget(QLabel("术前距牙槽嵴顶端2mm处的牙槽骨宽度(A值)："))
        self.A_value = QLabel("3.24")
        self.A_value.setStyleSheet("font-weight:bold;")
        self.A_layout.addWidget(self.A_value)
        self.A_layout.addWidget(QLabel("mm"))
        self.A_layout.addStretch(100)
        self.viewer_layout.addWidget(self.A_widget, stretch=1)
        # 穿过种植体的轴线在术前矢状面上测得的残余牙槽嵴高度(B值)
        self.B_widget = QWidget()
        self.B_layout = QHBoxLayout(self.B_widget)
        self.B_layout.addWidget(QLabel("穿过种植体的轴线在术前矢状面上测得的残余牙槽嵴高度(B值)："))
        self.B_value = QLabel("15.7")
        self.B_value.setStyleSheet("font-weight:bold;")
        self.B_layout.addWidget(self.B_value)
        self.B_layout.addWidget(QLabel("mm"))
        self.B_layout.addStretch(100)
        self.viewer_layout.addWidget(self.B_widget, stretch=1)
        # 穿过种植体的轴线在术后矢状面上测得的残余牙槽嵴高度(K值)
        self.K_widget = QWidget()
        self.K_layout = QHBoxLayout(self.K_widget)
        self.K_layout.addWidget(QLabel("穿过种植体的轴线在术后矢状面上测得的残余牙槽嵴高度(K值)："))
        self.K_value = QLabel("18.18")
        self.K_value.setStyleSheet("font-weight:bold;")
        self.K_layout.addWidget(self.K_value)
        self.K_layout.addWidget(QLabel("mm"))
        self.K_layout.addStretch(100)
        self.viewer_layout.addWidget(self.K_widget, stretch=1)
        # 种植体肩部上方唇骨的高度(CD)
        self.CD_widget = QWidget()
        self.CD_layout = QHBoxLayout(self.CD_widget)
        self.CD_layout.addWidget(QLabel("种植体肩部上方唇骨的高度(CD)："))
        self.CD_value = QLabel("1.44")
        self.CD_value.setStyleSheet("font-weight:bold;")
        self.CD_layout.addWidget(self.CD_value)
        self.CD_layout.addWidget(QLabel("mm"))
        self.CD_layout.addStretch(100)
        self.viewer_layout.addWidget(self.CD_widget, stretch=1)
        # 种植体平台处的牙槽骨宽度(EF)
        self.EF_widget = QWidget()
        self.EF_layout = QHBoxLayout(self.EF_widget)
        self.EF_layout.addWidget(QLabel("种植体平台处的牙槽骨宽度(EF)："))
        self.EF_value = QLabel("6.66")
        self.EF_value.setStyleSheet("font-weight:bold;")
        self.EF_layout.addWidget(self.EF_value)
        self.EF_layout.addWidget(QLabel("mm"))
        self.EF_layout.addStretch(100)
        self.viewer_layout.addWidget(self.EF_widget, stretch=1)
        # 种植体平台处的唇骨宽度(CE)
        self.CE_widget = QWidget()
        self.CE_layout = QHBoxLayout(self.CE_widget)
        self.CE_layout.addWidget(QLabel("种植体平台处的唇骨宽度(CE)："))
        self.CE_value = QLabel("2.13")
        self.CE_value.setStyleSheet("font-weight:bold;")
        self.CE_layout.addWidget(self.CE_value)
        self.CE_layout.addWidget(QLabel("mm"))
        self.CE_layout.addStretch(100)
        self.viewer_layout.addWidget(self.CE_widget, stretch=1)
        # 距种植体平台顶端2mm处的牙槽骨宽度(IJ)
        self.IJ_widget = QWidget()
        self.IJ_layout = QHBoxLayout(self.IJ_widget)
        self.IJ_layout.addWidget(QLabel("距种植体平台顶端2mm处的牙槽骨宽度(IJ)："))
        self.IJ_value = QLabel("8.19")
        self.IJ_value.setStyleSheet("font-weight:bold;")
        self.IJ_layout.addWidget(self.IJ_value)
        self.IJ_layout.addWidget(QLabel("mm"))
        self.IJ_layout.addStretch(100)
        self.viewer_layout.addWidget(self.IJ_widget, stretch=1)
        # 距种植体平台顶端2mm处的唇骨宽度(IG)
        self.IG_widget = QWidget()
        self.IG_layout = QHBoxLayout(self.IG_widget)
        self.IG_layout.addWidget(QLabel("距种植体平台顶端2mm处的唇骨宽度(IG)："))
        self.IG_value = QLabel("2.74")
        self.IG_value.setStyleSheet("font-weight:bold;")
        self.IG_layout.addWidget(self.IG_value)
        self.IG_layout.addWidget(QLabel("mm"))
        self.IG_layout.addStretch(100)
        self.viewer_layout.addWidget(self.IG_widget, stretch=1)

        # 部件主界面底部留白
        self.viewer_layout.addStretch(2)







