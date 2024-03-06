# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/03 16:58
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from PyQt5.QtWidgets import QMainWindow, QDialog

from designer_ui.RegistWindow import Ui_RegistWindow


class RegistWindow(QDialog, Ui_RegistWindow):

    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        # 渲染注册界面
        self.setupUi(self)
