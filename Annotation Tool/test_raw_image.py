import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2


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
    ceil = np.percentile(image, 70)  # 5% of pixels will be white. As the number increases,no. of white pixels increases
    floor = np.percentile(image, 10)  # 5% of pixels will be black
    return scale_image_base(image, ceil, floor)


def plot_stuff(raw_img, processed_img, cmap='gray'):
    images = [raw_img, processed_img]
    titles = ['raw_image', 'processed_image']  # In order to avoid this return the dict from funcs with key as name
    fig, ax = plt.subplots(nrows=len(images), ncols=1)
    for idx, val in enumerate(ax):
        val.imshow(images[idx], cmap=cmap)
        val.set_title(titles[idx])
    plt.show()


if __name__ == '__main__':
    raw_image = r'ToYash\xl_visual_00001490.bin'
    img = read_matrix(raw_image)
    print(img.shape)
    # plt.imsave('img.png', img)
    # new_img = np.fromfile('img.png', dtype='<d')
    # print(f'New Img: {new_img.shape}')
    out = scale_image(img)
    print(f'Shape of Scale Image: {out.shape}')
    plot_stuff(img, out)

