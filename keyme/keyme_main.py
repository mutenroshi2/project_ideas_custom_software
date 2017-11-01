#import necessary libraries
from PySide.QtGui import *
from PySide.QtCore import *
import keyme_gui
import about_main
import sys
import cv2
import numpy as np
import threading
import qimage2ndarray
import queue
import os
import imutils

class MainWindow(QMainWindow,keyme_gui.Ui_MainWindow):
    data_available = Signal()
    data_queue = queue.Queue()

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.video_size = QSize(800,600)
        self.stop_event = threading.Event()
        self.connect(self.start_streaming,SIGNAL("clicked()"),self.setup_camera)
        self.connect(self.img_capture,SIGNAL("clicked()"),self.image_cap_handler)
        self.data_available.connect(self.data_queue_manager)
        self.connect(self.img_browse,SIGNAL("clicked()"),self.browse_folder)
        self.connect(self.browse_save_path,SIGNAL("clicked()"),self.browse_path)
        self.connect(self.actionAbout, SIGNAL("triggered()"), self.about_gui)
        self.transparent_contours = cv2.imread('roi.png')
        self.stop_event = threading.Event()

    def about_gui(self):
        self.launch_gui_about = about_main.MainWindow()
        self.launch_gui_about.show()

    def emit_data_avail(self):
        self.data_available.emit()

    def data_queue_manager(self):
        if self.data_queue.empty() is False:
            data = self.data_queue.get()
            if data[0] == 'Raw Image':
                self.raw_image = data[1]
                self.timer.stop()
                self.capture.release()
                self.display_image(data[1])

            if data[0] == 'Saved Image':
                self.raw_image = data[1]
                self.display_image(data[1])
                try:
                    self.timer.stop()
                    self.capture.release()
                except:
                    pass

            if data[0] == 'Processed Image':
                self.display_image((data[1]))

    def setup_camera(self):
        """Initialize camera.
        """
        self.capture = cv2.VideoCapture(0)
        # print(self.capture.shape)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)

    def display_video_stream(self):
        ret, frame = self.capture.read()
        colorRanges = [
            ((19, 98, 110), (255, 255, 255), 'reflected/metal surface'),
            ((0, 0, 180), (255, 255, 255), 'reflected/metal surface')]

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            # frame = cv2.flip(frame, 1)
            for (lower, upper, colorName) in colorRanges:
                mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)

                (_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
                if len(cnts) > 0:
                    # find the largest contour in the mask, then use it to compute
                    # the minimum enclosing circle and centroid
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    (cX, cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    if radius > 10:
                        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                        cv2.putText(frame, colorName, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 255, 255), 2)

            width,height,_ = self.transparent_contours.shape
            # print(width,height,self.video_size.width(),self.video_size.height())
            x_offset = (self.video_size.width() // 2) - (width // 4)
            y_offset = (self.video_size.height() // 2) - (height // 1)
            for c in range(0,2):
                frame[y_offset:y_offset + self.transparent_contours.shape[0],
                x_offset:x_offset + self.transparent_contours.shape[1], c] = \
                self.transparent_contours[:, :, c] * (self.transparent_contours[:, :,2] / 180.0) + \
                frame[y_offset:y_offset + self.transparent_contours.shape[0],x_offset:x_offset +
                self.transparent_contours.shape[1], c] * (1.0 - self.transparent_contours[:, :, 2] / 180.0)
        image = qimage2ndarray.array2qimage(frame)
        self.video_disp.setPixmap(QPixmap.fromImage(image))

    def stop_streaming(self):
        try:
            self.timer.stop()
            self.capture.release()
        except AttributeError:
            pass

    def image_cap_handler(self):
        t = threading.Thread(target=self.image_capture)
        t.start()
        # t.join()
        # t = threading.Thread(target=self.image_processing, args=(self.raw_image,))
        # t.start()

    def image_capture(self):
        if(self.capture.isOpened()):
            # print('This is working')
            for i in range(2):
                ret,frame = self.capture.read()

        if self.file_name.text() != '':
            name = str(self.file_name.text())
        else:
            name = 'key_image'

        if self.select_folder != '':
            path = self.select_folder
        else:
            path = os.getcwd()

        cv2.imwrite(path + "\\" + name +".jpg",frame)
        self.data_queue.put(['Raw Image',frame])
        self.emit_data_avail()

    def image_processing(self,img):
        width, height, depth = img.shape
        img_copy = img.copy()
        # img_small = cv2.resize(img_copy, (
        #             img_copy.shape[1] // scale_factor, img_copy.shape[0] // scale_factor))
        # img_small = imutils.resize(img_copy,height=600)
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(~thresh, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(thresh, ret, ret * 0.9)
        (cx, cy) = (img_copy.shape[1] // 2, img_copy.shape[0] // 2)
        (_, cnts, _) = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # c = max(cnts,key= cv2.contourArea)
        # print(c)
        # print(cnts)
        cv2.drawContours(img_copy, cnts, -1, (0, 255, 0), 2)
        self.data_queue.put(['Processed Image', img_copy])
        self.emit_data_avail()

    def display_image(self,img):
        width = 800
        height = 600
        img_copy = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        img_copy = imutils.resize(img_copy, height=600)
        image = qimage2ndarray.array2qimage(img_copy)
        self.video_disp.setPixmap(QPixmap.fromImage(image))

    def rotate_image(self,img):
        rotated = imutils.rotate(img,10)
        return rotated

    def browse_folder(self):
        # self.stop_streaming()
        self.selected_directory, _ = QFileDialog.getOpenFileName(self, 'Open File', '',
                                                                 'Images (*.png *.jpg)', None,
                                                                 QFileDialog.DontUseNativeDialog)
        if self.selected_directory != '':
            img = cv2.imread(self.selected_directory)
            self.data_queue.put(['Saved Image',img])
            self.emit_data_avail()

        t = threading.Thread(target=self.image_processing, args=(self.raw_image,))
        t.start()

    def browse_path(self):
        self.select_folder = QFileDialog.getExistingDirectory(self, None, options=QFileDialog.DontUseNativeDialog)
        self.select_folder = self.select_folder.replace('/', '\\')
        self.browse_folder_path.setText(self.select_folder)

    def closeEvent(self, *args, **kwargs):
        self.hide()
        self.stop_event.set()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())