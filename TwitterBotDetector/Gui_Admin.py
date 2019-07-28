import ctypes
from datetime import datetime
from PyQt5.QtGui import QIcon, QPixmap,QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox, QFileDialog
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, QFile, QTextStream
from PyQt5 import QtCore
#from ModelTrainingUtils.CNNCreator import CNNCreator as CNNCreator
import pyqtgraph as pg
import multiprocessing as mp
from AccuracyHistory import AccuracyHistory, GraphIndex
from CNNThreadWork import CNNThreadWork
from Help_Window import Help_Window
import os

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from ManagerModule import ManagerModule
import time
import threading
import sys


class Feature:
    """
    class for better understanding indexes of the dictionary
    """
    BATCH_SIZE = 0
    EPOCH_NBR = 1
    MODEL_NAME = 2
    FEATURE_NBR = 3

    ACC_EPOCH  = 0
    LOSS_EPOCH = 1
    ACC_BATCH  = 2
    LOSS_BATCH = 3


class Gui_Admin(QWidget):
    """
    UI Class gives the ability to create and control training model phase
    """

    # py signals to communicate with main thread
    logText = QtCore.pyqtSignal([str, bool], [str])
    showMessageBox = QtCore.pyqtSignal(str)
    draw_plot = QtCore.pyqtSignal(pg.PlotWidget, list, list)

    def __init__(self,parent=None):
        super(Gui_Admin, self).__init__(parent)
        self.show
        self.queue = mp.Queue()
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        # init the initial parameters of this GUI
        self.title = 'Lie Detection'
        self.width = w
        self.height = h
        self.CNN = None
        self.title = 'Lie Detection - Admin'
        self._initModelDefaultParams()
        self._initUI()
        # connecting py signals
        self.showMessageBox.connect(self.on_show_message_box)
        self.draw_plot.connect(self.draw_graph)
        self.logText[str].connect(self._show_log)
        self.logText[str, bool].connect(self._show_log)
        self.CNNThread = None
        self.graph_arr = []


    def draw_graph(self,graph, y, x):
        """
        redrawing graph of learning phase
        :param graph: graph to draw
        :param y: data for y Axis
        :param x: data for x Axis
        """
        graph.clear()
        graph.plot(x, y, pen=pg.mkPen('r', width=5))
        QApplication.processEvents()

    @QtCore.pyqtSlot(str)
    @QtCore.pyqtSlot(str, bool)
    def _show_log(self, log_text, set_disable=False):
        """
        displaying log in UI
        :param log_text:
        :return:
        """
        if(set_disable == True):
            self.btnStartLearnPhase.setDisabled(True)

        self.text_edit.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S')+" : " + log_text)

    def _initModelDefaultParams(self):
        """
        initiate default parameters for model
        :return:
        """
        self.defaultDict = {'Batch size': 50 , 'Epoch Number': 10, 'Model Name': 'arcii_first_version_with_two_inputs'}
        self.comboText = 'adam'
        self.train_percent = 0.7

    def _initUI(self):
        """
        init UI
        """
        file = QFile(':css/StyleSheet.css')
        file.open(QFile.ReadOnly)
        stream = QTextStream(file)
        text = stream.readAll()
        self.setStyleSheet(text)

        self.setWindowIcon(QIcon(':Pictures/logo.png'))
        self.setWindowTitle(self.title)
        self.setGeometry(0, 0, self.width, self.height -60)
        

        # create main frame
        main_frame = QtWidgets.QFrame(self)
        main_layout = QtWidgets.QGridLayout(main_frame)
        main_frame.setFixedSize(self.width, self.height)
        main_frame.setObjectName("MainFrame")

        


        # Return to main window button
        self.returnBtn = QtWidgets.QPushButton("")
        self.returnBtn.setStyleSheet("QPushButton {background: url(:Pictures/backimg.png) no-repeat transparent;} ")
        self.returnBtn.setFixedWidth(110)
        self.returnBtn.setFixedHeight(110)
        self.returnBtn.clicked.connect(self.closeThisWindow)

        # help button
        helpBtn = QtWidgets.QPushButton("")
        helpBtn.setStyleSheet("QPushButton {background: url(:Pictures/help.png) no-repeat transparent;} ")
        helpBtn.setFixedWidth(110)
        helpBtn.setFixedHeight(110)
        helpBtn.clicked.connect(self.showHelp)
        buttonsform = QtWidgets.QFormLayout(self)

        buttonsform.addRow(self.returnBtn, helpBtn)
        
        # title frame
        title_frame = QtWidgets.QFrame()
        title_layout = QtWidgets.QHBoxLayout(title_frame)
        title_layout.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_frame,0,1,1,1)
        
        # form label
        formTitleLbl = QtWidgets.QLabel('Admin Management Settings')
        formTitleLbl.setObjectName("LableHeader")
        title_layout.addWidget(formTitleLbl)
        

        # sub frame with parameters for model
        first_sub_frame = QtWidgets.QFrame(main_frame)
        first_sub_layout = QtWidgets.QVBoxLayout(first_sub_frame)
        first_sub_frame.setFixedWidth(self.width/3)
        first_sub_frame.setFixedHeight(self.height/1.2)
        main_layout.addWidget(first_sub_frame,1,0,1,2)
        form_frame = QtWidgets.QFrame(first_sub_frame)
        form_frame.setObjectName("FormFrame")
        form_frame.setFixedHeight(self.height/3.5)
        form_frame.setFixedWidth(first_sub_frame.width()*0.9)
        form_layout = QtWidgets.QFormLayout(form_frame)
        lbl = QtWidgets.QLabel("Parameters")
        lbl.setStyleSheet("font-size:30px bold")
        lbl.setAlignment(Qt.AlignCenter)
        first_sub_layout.addWidget(lbl)
        first_sub_layout.addWidget(form_frame)

        # Graph frame


        self.graph_frame = QtWidgets.QFrame(main_frame)
        self.graph_frame.setFixedWidth(self.width*2/3)
        self.graph_frame.setFixedHeight(self.height*0.8)
        self.graph_layout = QtWidgets.QGridLayout(self.graph_frame)
        #self.graph_layout.setAlignment(Qt.AlignTop|Qt.AlignCenter)


        logo = QtWidgets.QLabel('', self)
        pixmap = QPixmap(':Pictures/logo.png')
        logo.setPixmap(pixmap)
        logo.setAlignment(Qt.AlignCenter)
        self.graph_layout.addWidget(logo)
        
    
        main_layout.addWidget(self.graph_frame, 1, 1, 2, 2)

        # Font for form label
        myFont = QtGui.QFont()
        myFont.setBold(True)
        myFont.setPixelSize(25)
        formTitleLbl.setFont(myFont)
        self.arrTxt = []
        arrLbl = []
        i = 0

        # create form with user parameters
        for key, value in self.defaultDict.items():
            self.arrTxt.append(QtWidgets.QTextEdit("" + str(value), self))
            if(key == 'Model Name'):
                self.arrTxt[i].setFixedWidth(250)
                self.arrTxt[i].setFixedHeight(25)
            else:
                self.arrTxt[i].setFixedWidth(50)
                self.arrTxt[i].setFixedHeight(25)
            self.arrTxt[i].setAlignment(Qt.AlignCenter)
            arrLbl.append(QtWidgets.QLabel(key, self))
            form_layout.addRow(arrLbl[i], self.arrTxt[i])
            i += 1

        # create combobox with optimizers
        #lblComboBox, self.comboBox = self._initCombobox()
        #form_layout.addRow(lblComboBox,self.comboBox)
        train_percentLbl, self.train_percentScale = self._initSlider()
        form_layout.addRow(train_percentLbl)
        form_layout.addRow(self.train_percentScale)

        # create button for start learning phase
        btnLayout = QtWidgets.QHBoxLayout()
        self.btnStartLearnPhase = QtWidgets.QPushButton("Start")
        self.btnStartLearnPhase.setFixedWidth(150)
        self.btnStartLearnPhase.setObjectName("Buttons")
        btnLayout.addWidget(self.btnStartLearnPhase)
        first_sub_layout.addLayout(btnLayout)
        self.btnStartLearnPhase.clicked.connect(lambda: self.learnPhase())

        # create log print window
        myFont = QtGui.QFont()
        myFont.setPixelSize(16)
        self.text_edit = QtWidgets.QTextEdit("")
        self.text_edit.setStyleSheet("color:black")
        self.text_edit.setFont(myFont)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFixedWidth(first_sub_frame.width()*0.9)
        self.text_edit.setFixedHeight(first_sub_frame.height()*0.45)
        lbl = QtWidgets.QLabel("Log:")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-size:32px bold")
        first_sub_layout.addWidget(lbl)
        first_sub_layout.addWidget(self.text_edit)
        self.showMaximized()

    def _initSlider(self):
        """
        creating slider for choosing percent data for training and validating option
        """
        train_percentLbl = QtWidgets.QLabel('Training percent =  {}%\t'.format(int(self.train_percent*100)))
        train_percentScale = QtWidgets.QSlider(Qt.Horizontal)
        train_percentScale.setFixedWidth(150)
        train_percentScale.setMinimum(0)
        train_percentScale.setMaximum(100)
        train_percentScale.setTickInterval(1)
        train_percentScale.setValue(80)
        train_percentScale.valueChanged.connect(lambda: self.updateSlideValue(train_percentScale, train_percentLbl))
        return train_percentLbl, train_percentScale

    def _initCombobox(self):
        """
        creating combobox with parameters
        """
        comboBox = QtWidgets.QComboBox(self)
        comboBox.addItem("adam")
        comboBox.addItem("sgd")
        comboBox.addItem("rmsprop")
        comboBox.setFixedWidth(150)
        comboBox.activated[str].connect(self.onActivated)
        comboBoxLbl = QtWidgets.QLabel('Optimizer')
        return comboBoxLbl, comboBox

    def onActivated(self, text):
        """
        updating textbox changed value
        """
        self.comboText = text

    def updateSlideValue(self, learnRateScale, train_percentLbl):
        """
        updating slide value change
        """
        self.train_percent = learnRateScale.value() / 100
        train_percentLbl.setText('Training percent =  ' + str(learnRateScale.value()) + '%')

    def learnPhase(self):
        """
        starting learning phase
        Batch size  =  0 < x < infinty (int)
        Epoch number = 0 < x < infinty (int)
        """
        if self.btnStartLearnPhase.text() == "Start":

            exceptionMsg = ""
            label = 0

            try:
                batch_size = int(self.arrTxt[Feature.BATCH_SIZE].toPlainText())
            except Exception as e:
                label = 1
                exceptionMsg = exceptionMsg + "Please enter only numbers to batch size!\n"
            try:
                epoch_nbr = int(self.arrTxt[Feature.EPOCH_NBR].toPlainText())
            except Exception as e:
                label = 1
                exceptionMsg = exceptionMsg + "Please enter only numbers to epoch number!\n"

            try:
                model_name = str(self.arrTxt[Feature.MODEL_NAME].toPlainText())
            except Exception as e:
                label = 1
                exceptionMsg = exceptionMsg + "Please enter again model name"
            if(label ==0):
                if batch_size <= 1 or batch_size > 1000:
                    exceptionMsg = exceptionMsg + "Batch size must be between 2 and 1000.\n"
                if epoch_nbr <= 1:
                    exceptionMsg = exceptionMsg + "Epoch number must be positive number bigger than 1.\n"
            if len(exceptionMsg) > 0:
                #QMessageBox.information(self, "Warning", exceptionMsg)
                mb = QMessageBox()
                mb.setIcon(QMessageBox.Information)
                mb.setWindowTitle('Warning')
                mb.setText(exceptionMsg)
                mb.setStandardButtons(QMessageBox.Ok)
                mb.exec_()

            else:
                    graphNames = ["Accuracy Epoch",
                                    "Loss Epoch",
                                    "Accuracy Batch",
                                    "Loss Batch"]
                    j = 0
                    pg.setConfigOption("antialias", True)
                    for i in range(4):
                        # Building the graphs
                        self.graph_arr.append(pg.PlotWidget(title=graphNames[i]))
                        self.graph_arr[i].showGrid(x=True, y=True)
                        self.graph_arr[i].getAxis('bottom').enableAutoSIPrefix(False)
                        self.graph_arr[i].getAxis('left').enableAutoSIPrefix(False)
                        self.graph_arr[i].setEnabled(False)
                        self.graph_layout.addWidget(self.graph_arr[i], j, i % 2, 1, 1)
                        if i == 1:
                            j += 1

                    # training model
                    self.btnStartLearnPhase.setText("Cancel")
                    self.init_graph_by_params()
                    displayGraph = AccuracyHistory(self.graph_arr,self.graph_frame,self.logText, self.draw_plot,epoch_nbr)
                    self.CNN_model = ManagerModule(train_perc=self.train_percent, batch_size=batch_size, epoch_nbr=epoch_nbr,model_name = model_name, calback_func =displayGraph, output=self.logText)
                    self.CNNThread = CNNThreadWork(self,self.CNN_model)
                    self.CNNThread.daemon = True
                    self.CNNThread.start()
                    self.text_edit.setText("")
                    self.changeDisable(True)


        else:
            QMessageBox.information(self, "Model train stopped", "Your model training stopped")
            self.CNNThread.stopThread()
            self.CNNThread.join()
            self.btnStartLearnPhase.setText("Start")
            self.graph_frame.setVisible(False)
            self.btnStartLearnPhase.setDisabled(False)
            self.changeDisable(False)



    def changeDisable(self,status):
        """
        change enable form manage when starting to train model or when finish to train model
        :param status:
        :return:
        """
        for txt in self.arrTxt:
            txt.setDisabled(status)
        self.returnBtn.setDisabled(status)
        self.train_percentScale.setDisabled(status)
        self.graph_frame.setVisible(status)

    def init_graph_by_params(self):
        """
        init graph labels
        """
        self.graph_arr[Feature.ACC_EPOCH].setLabel('bottom', 'Epoch number', units='times')
        self.graph_arr[Feature.ACC_EPOCH].setLabel('left', 'Accuracy', units='%')
        self.graph_arr[Feature.LOSS_EPOCH].setLabel('left', 'Loss value', units='%')
        self.graph_arr[Feature.LOSS_EPOCH].setLabel('bottom', 'Epoch number', units='times')
        self.graph_arr[Feature.ACC_BATCH].setLabel('bottom', 'Batch number', units='times')
        self.graph_arr[Feature.ACC_BATCH].setLabel('left', 'Accuracy', units='%')
        self.graph_arr[Feature.LOSS_BATCH].setLabel('left', 'Loss value', units='%')
        self.graph_arr[Feature.LOSS_BATCH].setLabel('bottom', 'Batch number', units='times')

    def on_show_message_box(self, res):
        """
        display messagebox at the end of the train
        """
        if res == 'Finished':
            mb = QMessageBox()
            mb.setIcon(QMessageBox.Information)
            mb.setWindowTitle('Info')
            mb.setText("The model create Succefuly.")
            mb.setStandardButtons(QMessageBox.Ok)
            mb.exec_()
        self.btnStartLearnPhase.setText("Start")
        self.graph_frame.setVisible(False)
        self.btnStartLearnPhase.setDisabled(False)
        self.changeDisable(False)
        for i in range(4):
            self.graph_arr[i].clear()
        self.graph_arr = []
        
    def file_save(self):
        """
        store the file with the name
        """
        #path, _ = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
        self.CNN_model.saveModel()
        #return path

    def closeThisWindow(self):
        """
        close current window
        """
        self.parent().show()
        self.parent().main_frame.setVisible(True)
        self.close()
    # Opens help window
    def showHelp(self):
        """
        display help
        """
        helpWindow = Help_Window(':Pictures/helpadmin3.png')
