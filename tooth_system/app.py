import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

from my_ui import LoginWindow, RegistWindow, EmailRetrievalWindow, PhoneRetrievalWindow, MainWindow


class ToothAnalysisSystem(object):
    def __init__(self):
        # 初始化登录、注册、邮件找回和电话找回等窗口
        self.login_window = LoginWindow()
        self.regist_window = RegistWindow()
        self.email_retrieval_window = EmailRetrievalWindow()
        self.phone_retrieval_window = PhoneRetrievalWindow()
        self.main_window = MainWindow()
        # 绑定所有窗口之前的跳转关系
        self.bind_toggle_windows()

    def run(self):
        # self.login_window.show()
        self.main_window.show()

    def login_submit(self):
        # 登录转主窗口
        self.login_window.login_submit_btn.clicked.connect(self.login_window.close)
        self.login_window.login_submit_btn.clicked.connect(self.main_window.show)

    def regist_submit(self):
        # 注册转登录
        self.regist_window.regist_submit_btn.clicked.connect(self.regist_window.close)
        self.regist_window.regist_submit_btn.clicked.connect(self.login_window.show)

    def email_submit(self):
        # 邮箱验证码修改密码转登录
        self.email_retrieval_window.submit_btn.clicked.connect(self.email_retrieval_window.close)
        self.email_retrieval_window.submit_btn.clicked.connect(self.login_window.show)

    def phone_submit(self):
        # 短信验证码修改密码转登录
        self.phone_retrieval_window.submit_btn.clicked.connect(self.phone_retrieval_window.close)
        self.phone_retrieval_window.submit_btn.clicked.connect(self.login_window.show)

    def bind_toggle_windows(self):
        # 登录转注册
        self.login_window.regist_btn.clicked.connect(self.login_window.close)
        self.login_window.regist_btn.clicked.connect(self.regist_window.show)
        # 登录转邮箱验证码
        self.login_window.forget_btn.clicked.connect(self.login_window.close)
        self.login_window.forget_btn.clicked.connect(self.email_retrieval_window.show)
        # 注册转登录
        self.regist_window.return_login_btn.clicked.connect(self.regist_window.close)
        self.regist_window.return_login_btn.clicked.connect(self.login_window.show)
        # 注册转邮箱验证码
        self.regist_window.forget_btn.clicked.connect(self.regist_window.close)
        self.regist_window.forget_btn.clicked.connect(self.email_retrieval_window.show)
        # 邮箱验证码转登录
        self.email_retrieval_window.return_login_btn.clicked.connect(self.email_retrieval_window.close)
        self.email_retrieval_window.return_login_btn.clicked.connect(self.login_window.show)
        # 邮箱验证码转短信验证码
        self.email_retrieval_window.phone_btn.clicked.connect(self.email_retrieval_window.close)
        self.email_retrieval_window.phone_btn.clicked.connect(self.phone_retrieval_window.show)
        # 短信验证码转登录
        self.phone_retrieval_window.return_login_btn.clicked.connect(self.phone_retrieval_window.close)
        self.phone_retrieval_window.return_login_btn.clicked.connect(self.login_window.show)
        # 短信验证码转邮箱验证码
        self.phone_retrieval_window.email_btn.clicked.connect(self.phone_retrieval_window.close)
        self.phone_retrieval_window.email_btn.clicked.connect(self.email_retrieval_window.show)

        # 各窗口提交按钮跳转逻辑
        self.login_submit()
        self.regist_submit()
        self.email_submit()
        self.phone_submit()


if __name__ == '__main__':
    # 创建QApplication类的实例
    app = QApplication(sys.argv)
    # 初始化牙齿分析系统
    toothAnalysisSystem = ToothAnalysisSystem()
    # 启动运行
    toothAnalysisSystem.run()
    # 进入程序的主循环，并通过exit函数确保主循环安全结束
    sys.exit(app.exec_())
