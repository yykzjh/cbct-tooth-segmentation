# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RegistWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_RegistWindow(object):
    def setupUi(self, RegistWindow):
        RegistWindow.setObjectName("RegistWindow")
        RegistWindow.resize(413, 393)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        RegistWindow.setWindowIcon(icon)
        self.verticalLayoutWidget = QtWidgets.QWidget(RegistWindow)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(70, 40, 281, 321))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(5, 0, 5, 0)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, 10, -1, 10)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.username_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.username_label.setEnabled(True)
        self.username_label.setMinimumSize(QtCore.QSize(50, 0))
        self.username_label.setMaximumSize(QtCore.QSize(16777215, 40))
        self.username_label.setStyleSheet("")
        self.username_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.username_label.setObjectName("username_label")
        self.horizontalLayout.addWidget(self.username_label)
        self.username_text = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.username_text.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.username_text.setFont(font)
        self.username_text.setObjectName("username_text")
        self.horizontalLayout.addWidget(self.username_text)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_4.setContentsMargins(-1, 10, -1, 10)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.password_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.password_label.setMinimumSize(QtCore.QSize(50, 0))
        self.password_label.setMaximumSize(QtCore.QSize(16777215, 40))
        self.password_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.password_label.setObjectName("password_label")
        self.horizontalLayout_4.addWidget(self.password_label)
        self.password_text = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.password_text.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.password_text.setFont(font)
        self.password_text.setInputMethodHints(QtCore.Qt.ImhHiddenText|QtCore.Qt.ImhNoAutoUppercase|QtCore.Qt.ImhNoPredictiveText|QtCore.Qt.ImhSensitiveData)
        self.password_text.setInputMask("")
        self.password_text.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_text.setCursorMoveStyle(QtCore.Qt.VisualMoveStyle)
        self.password_text.setObjectName("password_text")
        self.horizontalLayout_4.addWidget(self.password_text)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_5.setContentsMargins(-1, 10, -1, 10)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.phone_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.phone_label.setMinimumSize(QtCore.QSize(50, 0))
        self.phone_label.setMaximumSize(QtCore.QSize(16777215, 40))
        self.phone_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.phone_label.setObjectName("phone_label")
        self.horizontalLayout_5.addWidget(self.phone_label)
        self.phone_text = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.phone_text.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.phone_text.setFont(font)
        self.phone_text.setInputMethodHints(QtCore.Qt.ImhNone)
        self.phone_text.setInputMask("")
        self.phone_text.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.phone_text.setCursorMoveStyle(QtCore.Qt.VisualMoveStyle)
        self.phone_text.setObjectName("phone_text")
        self.horizontalLayout_5.addWidget(self.phone_text)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_8.setContentsMargins(-1, 10, -1, 10)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.email_label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.email_label.setMinimumSize(QtCore.QSize(50, 0))
        self.email_label.setMaximumSize(QtCore.QSize(16777215, 40))
        self.email_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.email_label.setObjectName("email_label")
        self.horizontalLayout_8.addWidget(self.email_label)
        self.email_text = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.email_text.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.email_text.setFont(font)
        self.email_text.setInputMethodHints(QtCore.Qt.ImhNone)
        self.email_text.setInputMask("")
        self.email_text.setEchoMode(QtWidgets.QLineEdit.Normal)
        self.email_text.setCursorMoveStyle(QtCore.Qt.VisualMoveStyle)
        self.email_text.setObjectName("email_text")
        self.horizontalLayout_8.addWidget(self.email_text)
        self.verticalLayout.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setSpacing(1)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.return_login_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.return_login_btn.setMaximumSize(QtCore.QSize(200, 30))
        font = QtGui.QFont()
        font.setFamily("Algerian")
        self.return_login_btn.setFont(font)
        self.return_login_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.return_login_btn.setStyleSheet("background:transparent;\n"
"color:blue;")
        self.return_login_btn.setObjectName("return_login_btn")
        self.horizontalLayout_6.addWidget(self.return_login_btn)
        self.forget_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.forget_btn.setMaximumSize(QtCore.QSize(200, 30))
        font = QtGui.QFont()
        font.setFamily("Algerian")
        self.forget_btn.setFont(font)
        self.forget_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.forget_btn.setStyleSheet("background:transparent;\n"
"color:blue;")
        self.forget_btn.setObjectName("forget_btn")
        self.horizontalLayout_6.addWidget(self.forget_btn)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.regist_submit_btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.regist_submit_btn.setMinimumSize(QtCore.QSize(250, 0))
        self.regist_submit_btn.setMaximumSize(QtCore.QSize(250, 40))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.regist_submit_btn.setFont(font)
        self.regist_submit_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.regist_submit_btn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.regist_submit_btn.setStyleSheet("color:white;\n"
"border:2px;\n"
"border-radius:10px;\n"
"padding:4px 4px;\n"
"font-size:14pt;\n"
"font-weight:bold;\n"
"background:qlineargradient(x1:0, y1:0, x2:1, y2:0,stop:0 #497BF0,stop:1 #1FB6F6);\n"
"hover {\n"
"    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #60a3f7, stop:1 #3b78e6);\n"
"}")
        self.regist_submit_btn.setObjectName("regist_submit_btn")
        self.horizontalLayout_7.addWidget(self.regist_submit_btn)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.verticalLayout.setStretch(0, 4)
        self.verticalLayout.setStretch(1, 4)
        self.verticalLayout.setStretch(2, 4)
        self.verticalLayout.setStretch(3, 4)
        self.verticalLayout.setStretch(4, 2)
        self.verticalLayout.setStretch(5, 4)

        self.retranslateUi(RegistWindow)
        QtCore.QMetaObject.connectSlotsByName(RegistWindow)

    def retranslateUi(self, RegistWindow):
        _translate = QtCore.QCoreApplication.translate
        RegistWindow.setWindowTitle(_translate("RegistWindow", "注册"))
        self.username_label.setText(_translate("RegistWindow", "用户名："))
        self.username_text.setPlaceholderText(_translate("RegistWindow", "请输入用户名"))
        self.password_label.setText(_translate("RegistWindow", "密码："))
        self.password_text.setPlaceholderText(_translate("RegistWindow", "请输入密码"))
        self.phone_label.setText(_translate("RegistWindow", "电话："))
        self.phone_text.setPlaceholderText(_translate("RegistWindow", "请输入电话号码"))
        self.email_label.setText(_translate("RegistWindow", "邮箱："))
        self.email_text.setPlaceholderText(_translate("RegistWindow", "请输入邮箱"))
        self.return_login_btn.setText(_translate("RegistWindow", "返回登录界面"))
        self.forget_btn.setText(_translate("RegistWindow", "忘记密码"))
        self.regist_submit_btn.setText(_translate("RegistWindow", "注册"))
