import numpy as np
import sys
from PyQt5.QtGui import QImage, qRgb
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
import cv2
#import matplotlib.pyplot as plt


def read_matrix(file_name):
    data = np.fromfile(file_name, dtype='<d')  # little-endian double precision float
    nr_rows = 512
    nr_cols = int(len(data) / nr_rows)
    img = data.reshape((nr_rows, nr_cols))
    return img


def scale_image_base(image, ceil, floor):
    a = 255 / (ceil - floor)
    b = floor * 255 / (floor - ceil)
    out = np.maximum(0, np.minimum(255, image * a + b)).astype(np.uint8)
    return out


def scale_image(image):
    ceil = np.percentile(image, 10)  # 5% of pixels will be white
    floor = np.percentile(image, 5)  # 5% of pixels will be black
    return scale_image_base(image, ceil, floor)


def numpyQImage(image):
    qImg = QImage()
    if image.dtype == np.uint8:
        if len(image.shape) == 2:
            channels = 1
            height, width = image.shape
            bytesPerLine = channels * width
            qImg = QImage(
                image.data, width, height, bytesPerLine, QImage.Format_Indexed8
            )
            qImg.setColorTable([qRgb(i, i, i) for i in range(256)])
        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                height, width, channels = image.shape
                bytesPerLine = channels * width
                qImg = QImage(
                    image.data, width, height, bytesPerLine, QImage.Format_RGB888
                )
            elif image.shape[2] == 4:
                height, width, channels = image.shape
                bytesPerLine = channels * width
                fmt = QImage.Format_ARGB32
                qImg = QImage(
                    image.data, width, height, bytesPerLine, QImage.Format_ARGB32
                )
    return qImg


def raw_to_QImage(raw_image):
    file = read_matrix(raw_image)
    # cv2.imshow('Image', file)
    out = scale_image(file)
    cv2.imshow('Image', out)
    cv2.imwrite('test.png', out)
    # new = out.copy()
    # image = numpyQImage(new)
    # pixmap = QPixmap.fromImage(image)
    # return image


if __name__ == '__main__':
    # app = QApplication(sys.argv)
    #
    # win = QWidget()
    # label = QLabel()

    raw_image = r'D:\FH-AACHEN\Thesis\Dataset\SignalImages\xl_signal_00000813.bin'
    # openimg = QImage.loadFromData(raw_image)

    raw_to_QImage(raw_image)
    # label.setPixmap(pixmap)
    #
    # vbox = QVBoxLayout()
    # vbox.addWidget(label)
    # win.setLayout(vbox)
    # win.show()
    # sys.exit(app.exec_())
