from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

class MetricsGui():
    def __init__(self):
        self.win = QMainWindow()
        self.win.setGeometry(0, 0, 300, 30)
        self.win.setWindowTitle("Metrics")
        self.curr_pos_dis = QtWidgets.QLabel(self.win)
        self.curr_pos_dis.setText("Testing")
        self.win.show()

    def refresh(self):
        self.curr_pos_dis.setText("Tested")
        self.win.show()
        