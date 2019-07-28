from matplotlib.figure import Figure
from PyQt5.QtGui import QPixmap, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QWidget
from PyQt5 import QtGui, QtWidgets
import time
import threading
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QFile, QTextStream
import pyqtgraph
import ctypes
from Help_Window import Help_Window
import sys
import os
from UserModule import UserModule


class Gui_User(QWidget):
    def __init__(self, parent=None):
        super(Gui_User, self).__init__(parent)
        # init the initial parameters of this GUI
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        self.title = 'Twitter Bot Detector'
        self.width = w
        self.height = h
        self.startRec = True
        #stream params
        self.CHUNK = 1024
        self.CHANNELS = 2
        self.RATE = 8000
        self.frames = None
        self.pyrecorded = None
        self.stream = None
        self.recThread = None
        self.movie = None
        self.figureSoundWav = None
        self.mfccResult = None
        self.TXT_OUTPUT_FILENAME = None
        self.TXT_OUTPUT_FILEPATH = None
        self.pickedModelPath = None
        self.modelname = None
        self.checkEnv = True
        self.checkEnvErr = ""
        self.initUI()
        self.ret = ""



    def initUI(self):
        file = QFile(':css/StyleSheet.css')
        file.open(QFile.ReadOnly)
        stream = QTextStream(file)
        text = stream.readAll()
        self.setStyleSheet(text)
        self.setObjectName("Windowimg")
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon(':Pictures/logo.png'))
        self.setGeometry(0, 0, self.width, self.height-60)
        #Creating main container-frame, parent it to QWindow
        self.main_frame = QtWidgets.QFrame(self)
        self.main_frame.setObjectName("MainFrame")
        self.main_frame.setFixedSize(self.width, self.height)


        #the first sub window
        main_layout = QtWidgets.QVBoxLayout(self.main_frame)
        self.firstsub_Frame = QtWidgets.QFrame(self.main_frame)

        main_layout.addWidget(self.firstsub_Frame)
        self.firstsub_Layout = QtWidgets.QFormLayout(self.firstsub_Frame)
        self.firstsub_Frame.setFixedHeight(self.height/5.2)

        # Return to main window button
        returnBtn = QtWidgets.QPushButton("")
        returnBtn.setStyleSheet("QPushButton {background: url(:Pictures/backimg.png) no-repeat transparent;} ")
        returnBtn.setFixedWidth(110)
        returnBtn.setFixedHeight(110)
        returnBtn.clicked.connect(self.closeThisWindow)

        # help button
        helpBtn = QtWidgets.QPushButton("")
        helpBtn.setStyleSheet("QPushButton {background: url(:Pictures/help.png) no-repeat transparent;} ")
        helpBtn.setFixedWidth(110)
        helpBtn.setFixedHeight(110)
        helpBtn.clicked.connect(self.showHelp)
        buttonsform = QtWidgets.QFormLayout(self)

        buttonsform.addRow(returnBtn, helpBtn)
        #Setting up the form fields
        #form title init
        formTitleLbl = QtWidgets.QLabel('Twitter Bot Detector')
        formTitleLbl.setAlignment(Qt.AlignCenter)
        formTitleLbl.setContentsMargins(0,0,50,50)
        formTitleLbl.setObjectName("LableHeader")
        self.firstsub_Layout.addRow(formTitleLbl)

        #init the browse file fields - lable , textfield, file browse button
        fileBrowseHBoxLayout = QtWidgets.QGridLayout()
        self.fileBrowserTxt=QtWidgets.QTextEdit("", self)
        self.fileBrowserTxt.setReadOnly(True)
        self.fileBrowserLbl=QtWidgets.QLabel('Pick txt File', self)
        
        self.fileBrowserTxt.setFixedWidth(500)
        self.fileBrowserTxt.setFixedHeight(25)
        self.fileBrowserLbl.setFixedWidth(150)
        self.fileBrowserLbl.setFixedHeight(25)
        self.fileBrowserBtn = QtWidgets.QPushButton("", self)
        self.fileBrowserBtn.setMaximumHeight(100)
        self.fileBrowserBtn.setMaximumWidth(100)
        self.fileBrowserBtn.setFixedHeight(27)
        self.fileBrowserBtn.setFixedWidth(27)
        self.fileBrowserBtn.setStyleSheet("QPushButton {background: url(:Pictures/filebrowse.png) no-repeat transparent;} ")
        self.fileBrowserBtn.clicked.connect(lambda: self.openFile(self.firstsub_Layout))
        fileBrowseHBoxLayout.addWidget(self.fileBrowserLbl,1,0)
        fileBrowseHBoxLayout.addWidget(self.fileBrowserTxt,1,1)
        fileBrowseHBoxLayout.addWidget(self.fileBrowserBtn,1,2)
        fileBrowseHBoxLayout.setAlignment(Qt.AlignCenter)
        self.firstsub_Layout.addRow(fileBrowseHBoxLayout)

        # Settings Layout
        self.settings_Frame = QtWidgets.QFrame(self.main_frame)
        main_layout.addWidget(self.settings_Frame)
        self.settings_Layout = QtWidgets.QFormLayout(self.settings_Frame)
        self.settings_Frame.setFixedWidth(self.width)
        self.settings_Frame.setFixedHeight(self.height/8)
        self.settings_Frame.setContentsMargins(self.width, 0, 0, 0)
        self.settings_Layout.setFormAlignment(Qt.AlignCenter)
        #self.settings_Frame.setVisible(False)
        # the third sub window
        self.thirdsub_Frame = QtWidgets.QFrame(self.main_frame)
        main_layout.addWidget(self.thirdsub_Frame)
        self.thirdsub_Layout = QtWidgets.QGridLayout(self.thirdsub_Frame)
        self.thirdsub_Frame.setFixedWidth(self.width-25)
        self.thirdsub_Frame.setFixedHeight(self.height/2.2)
        logo = QtWidgets.QLabel('', self)
        pixmap = QPixmap(':Pictures/logo.png')
        logo.setPixmap(pixmap)
        self.thirdsub_Layout.addWidget(logo)

        logo.setAlignment(Qt.AlignCenter|Qt.AlignTop)

        # building the Model comboBox
        self.buildModelComboBox()
        self.buildThreshold()


        #Predict button
        self.processGraphsBtn = QtWidgets.QPushButton("Predict", self)
        self.processGraphsBtn.setObjectName("pred_Buttons")
        self.processGraphsBtn.setFixedWidth(131)
        self.processGraphsBtn.setFixedHeight(30)
        self.processGraphsBtn.clicked.connect(lambda: self.dataProcessingModel())
        self.processGraphsBtn.setContentsMargins
        self.settings_Layout.addRow(self.processGraphsBtn)
        #self.processGraphsBtn.setAlignment(Qt.AlignCenter)


        #show the window
        self.show()


    def checkEnvironment(self,type):
        """
        Validate that the working environment is safe to work .

        """
        checkEnv = True
        self.checkEnvErr = ""
        winmm = ctypes.windll.winmm

        # Checking existing models
        modelPath = os.path.dirname(os.path.realpath(sys.argv[0])) + "\\Model\\"
        modelDir = os.listdir(modelPath)
        if len(modelDir) == 0:
            checkEnv = False
            self.checkEnvErr = self.checkEnvErr + "There is no Models to work with."

        return checkEnv

    def buildThreshold(self):
        self.comboBoxCoef = []
        self.comboBoxCoef.append(QtWidgets.QTextEdit("" + "0.5", self))
        self.comboBoxCoefLbl = QtWidgets.QLabel('Threshold')
        self.comboBoxCoefLbl.setFixedWidth(125)
        self.comboBoxCoefLbl.setFixedHeight(25)
        self.comboBoxCoef[0].setFixedWidth(130)
        self.comboBoxCoef[0].setFixedHeight(25)
        self.settings_Layout.addRow(self.comboBoxCoefLbl,self.comboBoxCoef[0])

    def buildModelComboBox(self):
        """
        Building the Model's combobox
        """
        self.comboboxModel = QtWidgets.QComboBox(self)
        self.comboboxModel.setFixedWidth(130)
        self.comboboxModel.setFixedHeight(25)
        self.comboboxModel.activated[str].connect(self.onActivatedComboBoxModel)
        modelPath = os.path.dirname(os.path.realpath(sys.argv[0])) + "\\Model\\"
        first = True
        for modelname in os.listdir(modelPath):
            if modelname.endswith('.h5'):
                self.comboboxModel.addItem(modelname.split('.')[0])
                if first:
                    self.pickedModelPath =modelPath +modelname
                    self.modelname = modelname
                    first = False
        self.comboBoxModelLbl = QtWidgets.QLabel('Model')
        self.comboBoxModelLbl.setFixedWidth(75)
        self.comboBoxModelLbl.setFixedHeight(25)
        self.settings_Layout.addRow(self.comboBoxModelLbl,self.comboboxModel)


    def onActivatedComboBoxModel(self, text):
        """
        Getting the Model once the user click on the Coefficients combobox
        :param text: The text that the user clicked on in the combobox
        """
        self.modelname = text
        self.pickedModelPath = os.path.dirname(os.path.realpath(sys.argv[0])) + "\\Model\\"+text+'.h5'


    def initSettings(self):
        """
        Initialize the settings before displaying graphs
        """
        self.clearGraph()

        #self.settings_Frame.setVisible(False)

    def openFile(self,form ):
        """
        Opening file browser to import the txt file.
        :param form: The current layout to display the message box error .
        """
        if self.checkEnvironment(2):
            self.initSettings()
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getOpenFileName(None, "File Browser", "", "Text Files (*.txt)", options=options)
            word = fileName.split('/')
            word = word[len(word) - 1]
            if len(word) != 0:
                if word.endswith('.txt'):
                    self.fileBrowserTxt.setText(''+word)
                    self.TXT_OUTPUT_FILENAME = word
                    self.TXT_OUTPUT_FILEPATH = fileName
                    self.dataProcessing()
                else:
                    QMessageBox.about(form, "Error", "Wrong file type , please use only txt files")
        else:
            QMessageBox.about(self, "Error", self.checkEnvErr)


    def dataProcessing(self):
        """
        Handiling the data processing.
        """
        # Showing te graph's frame.
        #self.settings_Frame.setVisible(True)
        #self.secondsub_Frame.setVisible(False)
        # Drawing the two graphs
        self.showGraphBig()
        self.showGraph()

    def dataProcessingModel(self):
        """
        Processing the txt file , drawing graph.
        """
        exceptionMsg = ""
        label = 0
        # Drawing mfcc for the input file.
        try:
            #setFocusPolicy(QT::WheelFocus)
            threshold = float(self.comboBoxCoef[0].toPlainText())
        except Exception as e:
           exceptionMsg = exceptionMsg + "Please enter only numbers to threshold!\n"
           label = 1
               
        if(label == 0):
            if (threshold < 0) or (threshold > 1):
                    exceptionMsg = exceptionMsg + "Threshold need to be between 0 to 1!\n"
        if(self.TXT_OUTPUT_FILEPATH is None):
            exceptionMsg = exceptionMsg + "Please choose predict file!\n"
        if(self.modelname is None):
            exceptionMsg = exceptionMsg + """Please put 'our_model' files in 'model' directory!"""
        if len(exceptionMsg) > 0:
            #QMessageBox.information(self, "Warning", exceptionMsg)
            mb = QMessageBox()
            mb.setIcon(QMessageBox.Information)
            mb.setWindowTitle('Warning')
            mb.setText(exceptionMsg)
            mb.setStandardButtons(QMessageBox.Ok)
            mb.exec_()
            #raise Exception(exceptionMsg)
        else:
            us = UserModule()
            res = us.predict(model_name=self.modelname,user_tweet=self.TXT_OUTPUT_FILEPATH, threshold= threshold)

            if(res == 1):
                #bot
                helpWindow = Help_Window(':Pictures/bot_message.png')
            elif(res ==0):
                #human
                helpWindow = Help_Window(':Pictures/human_message.png')
            #self.showGraph()
            # Prediction using the picked model .
            #check result bot or not.



    def clearGraph(self):
        """
        Clearing graphs
        :param layoutnum: the layout number that includes the wanted graph to clear.
        layoutnum = 3 -> the graph.
       

        while self.thirdsub_Layout.count():
            child = self.thirdsub_Layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                 """
        pass


    def showGraphBig(self ):
        """
        Drawing second graph.
        """
        pass

    def showGraph(self):
        """
        Drawing the graph.
        """
    pass


    def showHelp(self):
        """
        Opens help window.
        """
        helpWindow = Help_Window(':Pictures/helpuser3.png')


    def closeThisWindow(self):
        """
        Close the current window and open the main window.
        """
        self.parent().show()
        self.parent().main_frame.setVisible(True)
        self.close()