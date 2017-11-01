import about_gui
import sys
from PySide.QtGui import *
from PySide.QtCore import *

class MainWindow(QMainWindow,about_gui.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())