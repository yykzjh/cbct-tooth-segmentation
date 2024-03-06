# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/03/03 22:03
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from PyQt5 import QtGui, QtCore, QtWidgets


class HorizontalTabBar(QtWidgets.QTabBar):
    def paintEvent(self, event):

        painter = QtWidgets.QStylePainter(self)
        option = QtWidgets.QStyleOptionTab()
        for index in range(self.count()):
            self.initStyleOption(option, index)
            painter.drawControl(QtWidgets.QStyle.CE_TabBarTabShape, option)
            painter.drawText(self.tabRect(index),
                             QtCore.Qt.AlignCenter | QtCore.Qt.TextDontClip,
                             self.tabText(index))

    def tabSizeHint(self, index):
        size = QtWidgets.QTabBar.tabSizeHint(self, index)
        if size.width() < size.height():
            size.transpose()
        return size


class TabWidget(QtWidgets.QTabWidget):
    def __init__(self, parent=None):
        QtWidgets.QTabWidget.__init__(self, parent)
        self.setTabBar(HorizontalTabBar())


