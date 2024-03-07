# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/03 17:00
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QDialog

from designer_ui.EmailRetrievalWindow import Ui_EmailRetrievalWindow


class EmailRetrievalWindow(QDialog, Ui_EmailRetrievalWindow):

    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        # 渲染邮箱验证码修改密码界面
        self.setupUi(self)

        # 临时添加提示文字
        self.tip_email_label.setText("yyk****@163.com")
        self.tip_email_label.setStyleSheet(
            '''
            color:red;
            font-size:16px;
            '''
        )

        # 给发送验证码的按钮绑定事件处理函数
        self.send_code_btn.clicked.connect(self.send_code)
        # 初始化倒计时和定时器
        self.count_down_number = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.show_count_down)

    def send_code(self):
        self.send_code_btn.setEnabled(False)
        self.count_down_number = 60
        self.timer.start(1000)

    def show_count_down(self):
        # 在按钮上展示倒计时
        self.send_code_btn.setText(str(self.count_down_number) + "s")
        # 倒计时减少1并判断是否结束
        self.count_down_number -= 1
        if self.count_down_number < 0:
            self.timer.stop()
            self.send_code_btn.setText("发送")
            self.send_code_btn.setEnabled(True)
