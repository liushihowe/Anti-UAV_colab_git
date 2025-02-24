# import sys
# from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget
# from Ui_主界面 import Ui_MainWindow  #导入你写的界面类
 
 
# class MyMainWindow(QMainWindow, Ui_MainWindow): #这里也要记得改
#     def __init__(self,parent =None):
#         super(MyMainWindow, self).__init__(parent)
#         self.setupUi(self)
 
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     myWin = MyMainWindow()
#     myWin.show()
#     sys.exit(app.exec_())    

import sys
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QLineEdit
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import image

import cv2
import sqlite3
from ultralytics.models import YOLO
import cv2 

# ### 主页面设计，略
# class MainWindow(QMainWindow, mainwindow.Ui_MainWindow):
#     def __init__(self, parent=None):
#         super(MainWindow, self).__init__(parent)
#         self.setupUi(self)
#         self.image_open = Image_open()
#         self.pushButton.clicked.connect(self.image_open.show)
### 未详细说明，请参考前三篇博客
        
class Image_open(QMainWindow, image.Ui_MainWindow):
    def __init__(self, parent=None):
        super(Image_open, self).__init__(parent)
        # UI界面
        self.setupUi(self)
        self.background()
        self.cap = cv2.VideoCapture()
        self.num = 1
        self.playing = False
        # 在label中播放视频
        self.init_timer()
        
    def background(self):
 		# 文件选择按钮
        self.pushButton.clicked.connect(self.pre_judge)
        # 视频播放图标
        self.pushButton_2.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))  # 播放图标
        self.pushButton_3.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))  # 停止图标
        
        # 用于开始播放视频的按钮
        self.pushButton_2.clicked.connect(self.play_file) # 这是对应的函数
        # 用于关闭播放视频的按钮
        self.pushButton_3.clicked.connect(self.close_file) # 这是对应的函数
        
		# 当播放的为图像时，设计这两个按钮不能点击，只有当播放的是视频时，才能点击
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        
    def pre_judge(self):
        # 创建文件对话框，如果是视频，令self.video = True，如果是图像，令self.video = False，
        # 当self.video = None时，报错。
        self.video = None
        self.img_path = QFileDialog.getOpenFileName()[0]
        self.textBrowser.setText(self.img_path)
        video_type = [".mp4", ".mkv", ".MOV", "avi"]
        img_type = [".bmp", ".jpg", ".png", ".gif"]
        for vdi in video_type:
            if vdi not in self.img_path:
                continue
            else:
                self.video = True
                # 当是视频时，将开始按钮置为可点击状态
                self.pushButton_2.setEnabled(True)
        for ig in img_type:
            if ig not in self.img_path:
                continue
            else:
                self.video = False
                img = QPixmap(self.img_path)
                w = img.width()
                h = img.height()
                ratio = max(w / self.label.width(), h / self.label.height())
                img.setDevicePixelRatio(ratio)
                self.label.setAlignment(Qt.AlignCenter)
                self.label.setPixmap(img)

                self.label_2.setAlignment(Qt.AlignCenter)
                self.label_2.setPixmap(img)
        if self.video is None:
            QMessageBox.information(self, "警告", "我们暂时不支持此格式的文件！", QMessageBox.Ok)


    def play_file(self):
        self.label.setEnabled(True)
        # 如果播放视频，则使得关闭视频按钮可用
        self.pushButton_3.setEnabled(True)
        # 视频流阻塞信号关闭
        self.timer.blockSignals(False)

        # 如果计时器没激活，证明是暂停阶段，需要重新播放，并把self.playing = True。
        if self.timer.isActive() is False:
            self.cap.open(self.img_path)
            self.timer.start(30)
            self.playing = True
            # 更换播放按钮为暂停按钮
            self.set_state()
        # 如果计时器激活了，并且num为奇数，证明是播放阶段，需要暂停播放，并把self.playing = False。
        elif self.timer.isActive() is True and self.num % 2 == 1:
            self.timer.blockSignals(True)
            self.playing = False
            self.num = self.num + 1
            self.set_state()
        # 如果计时器激活了，并且num为偶数，证明经过播放阶段，现在是暂停阶段，需要重新开始播放，并把self.playing = True。
        elif self.timer.isActive() is True and self.num % 2 == 0:
            self.num = self.num + 1
            self.timer.blockSignals(False)
            self.playing = True
            self.set_state()
        else:
            QMessageBox.information(self, "警告", "视频播放错误！", QMessageBox.Ok)

    # 关闭本地视频
    def close_file(self):
        self.cap.release()
        self.pushButton_2.setEnabled(True)
        self.pushButton_3.setEnabled(False)
        self.timer.stop()
        self.playing = False
        # 关闭视频将按钮置为可以播放
        self.set_state()

    # 本地视频播放暂停转换图标按钮
    def set_state(self):
        if self.playing:
        	# 暂停图标
            self.pushButton_2.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

        else:
            self.pushButton_2.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
  
    # 播放视频画面
    def init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_pic)

    # 显示视频图像
    def show_pic(self):
        ret, img = self.cap.read()

        if ret:
            cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            model = YOLO("../runs/train_2025-02-05_08-29-53/weights/best.pt")
            results = model.predict(source=img, save=False, imgsz=320, conf=0.5) 

            # 视频流的长和宽
            height, width = cur_frame.shape[:2]
            # pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
            # pixmap = QPixmap.fromImage(pixmap)

            pixmap = QImage(results[0].plot(), width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)

            # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
            ratio = max(width/self.label.width(), height/self.label.height())
            pixmap.setDevicePixelRatio(ratio)
            # 视频流置于label中间部分播放
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setPixmap(pixmap)

            self.label_2.setAlignment(Qt.AlignCenter)
            self.label_2.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Image_open()
    main.show()
    sys.exit(app.exec_())
