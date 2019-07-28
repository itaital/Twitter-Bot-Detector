import ctypes

from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox

class Help_Window(QWidget):
    def __init__(self,imgpath, parent=None):
        super(Help_Window, self).__init__(parent)
        self.imgpath=imgpath
        # init the initial parameters of this GUI
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
        pixmap2 = QPixmap(imgpath)
        helpUser = QMessageBox()
        helpUser.setWindowTitle("Help")
        helpUser.setWindowIcon(QIcon(':Pictures/logo.png'))
        helpUser.setIconPixmap(pixmap2)
        helpUser.show()
        helpUser.exec()
