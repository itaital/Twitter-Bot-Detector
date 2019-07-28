from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QFile, QTextStream
#from ModelTrainingUtils.CNNCreatorimport *
import ctypes
from Gui_Admin import Gui_Admin
from Gui_User import Gui_User
from Help_Window import Help_Window
import sys
#don't delete using python files with image and css source
import design
import css
import os
#import xgboost
import sklearn.ensemble
import sklearn.tree
import pickle
import pandas as pd
import sklearn.neighbors.typedefs
import sklearn.neighbors.quad_tree
import sklearn.tree._utils
import cython
import sklearn
import sklearn.utils._cython_blas
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

class Main_Window(QWidget):
    def __init__(self, parent=None):
        super(Main_Window, self).__init__(parent)
        # init the initial parameters of this GUI
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        self.title = 'Twitter Bot Detector'
        self.width = w
        self.height = h
        self.initUI()



    def initUI(self):
        file = QFile(':css/StyleSheet.css')
        file.open(QFile.ReadOnly)
        stream = QTextStream(file)
        text = stream.readAll()
        self.setStyleSheet(text)
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon(":Pictures/logo.png"))
        self.setGeometry(0, 0, self.width, self.height-60)
        #Creating main container-frame, parent it to QWindow
        self.main_frame = QtWidgets.QFrame(self)
        self.main_frame.setObjectName("MainFrame")
        self.main_frame.setFixedWidth(self.width)
        self.main_frame.setFixedHeight(self.height)

        #the first sub window
        self.main_layout = QtWidgets.QVBoxLayout(self.main_frame)
        self.firstsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.firstsub_Frame.setFixedWidth(self.width)
        self.firstsub_Frame.setFixedHeight(400)
        self.main_layout.addWidget(self.firstsub_Frame)
        self.firstsub_Layout = QtWidgets.QHBoxLayout(self.firstsub_Frame)
        self.firstsub_Layout.setAlignment(Qt.AlignCenter)

        # help button
        helpBtn = QtWidgets.QPushButton("",self)
        helpBtn.setStyleSheet("QPushButton {background: url(:Pictures/help.png) no-repeat transparent;} ")
        helpBtn.setFixedWidth(110)
        helpBtn.setFixedHeight(110)
        helpBtn.clicked.connect(self.showHelp)

        # The second sub window
        self.secondsub_Frame = QtWidgets.QFrame(self.main_frame)
        self.main_layout.addWidget(self.secondsub_Frame)
        self.secondsub_Layout = QtWidgets.QHBoxLayout(self.secondsub_Frame)
        self.secondsub_Frame.setFixedWidth(self.width)
        self.secondsub_Layout.setAlignment(Qt.AlignTop|Qt.AlignCenter)

        #Setting up the fields

        logo = QtWidgets.QLabel('',self)
        pixmap = QPixmap(":Pictures/logo.png")
        logo.setPixmap(pixmap)
        self.firstsub_Layout.addWidget(logo)
        logo.setAlignment(Qt.AlignCenter)

        # Admin button
        adminBtn = QtWidgets.QPushButton("Settings", self)
        adminBtn.setObjectName("MainGuiButtons")
        adminBtn.clicked.connect(self.openAdminGui)

        # User button
        userBtn = QtWidgets.QPushButton("Analyze", self)
        userBtn.setObjectName("MainGuiButtons")
        userBtn.clicked.connect(self.openUserGui)

        self.secondsub_Layout.addWidget(adminBtn)
        self.secondsub_Layout.addWidget(userBtn)

        # Footer layout

        creditsLbl = QtWidgets.QLabel('Created By Itai Tal & Ariel Yaakov\n'
                                      'Supervisor: Zeev Vladimir Volkovich\n'
                                      '22/05/2019')
        creditsLbl.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(creditsLbl)



        #show the window
        self.showMaximized()


    def openAdminGui(self):
        adminGui = Gui_Admin(self)
        adminGui.show()
        self.main_frame.setVisible(False)

    def openUserGui(self):
        userGui = Gui_User(self)
        userGui.show()
        self.main_frame.setVisible(False)

    # Opens help window
    def showHelp(self):
        helpWindow = Help_Window(':Pictures/helpmain2.png')



if __name__ == '__main__':
    directory =os.path.dirname(sys.argv[0])
    if not os.path.exists(directory+"/Model"):
        os.mkdir(directory + "/Model")
    if not os.path.exists(directory + "/db"):
        os.mkdir(directory + "/db")
    if not os.path.exists(directory+"/logs"):
        os.mkdir(directory+"/logs")
    app = QApplication(sys.argv)
    main = Main_Window()
    sys.exit(app.exec_())