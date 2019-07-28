import threading
import time
from threading import Thread

from PyQt5 import QtCore
from PyQt5.QtCore import QThread


class CNNThreadWork(Thread):
    def __init__(self, app, CNN):
        super(CNNThreadWork, self).__init__()
        self.logger = app
        self.CNN_model = CNN
        self.is_run = False

    def stopThread(self):
        """
        stop thread if the user pressed stop train
        :return:
        """
        self.is_run = False
        self.CNN_model.set_running_status(self.is_run)

    def run(self):
        """
        Thread process training flow
        :return:
        """
        self.is_run = True
        self.CNN_model.set_running_status(self.is_run)
        time.sleep(2)
        # print to log with disable start button because train model is an atomic function
        # and there is no way to stop it except destroy the thread
        self.logger.logText[str, bool].emit("starting to train model...", True)
        if not self.is_run:
            return
        self.CNN_model.create_model()
        self.logger.logText[str].emit("Finished training model")
        if not self.is_run:
            return
        self.logger.showMessageBox[str].emit('Finished')
