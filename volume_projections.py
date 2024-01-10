import numpy as np
from PIL import Image
import os
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt


def open_image_from_desktop(image_name):
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    image_path = os.path.join(desktop_path, image_name)
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    return image


def binarize_image(image, threshold):
    binary_image = np.where(image > threshold, 1, 0)
    return binary_image


def count_white_pixels(image):
    white_pixels = np.sum(image)
    return white_pixels


if __name__ == '__main__':
        
    img_name = '747_PAG_20x_crop.tif'

    img = open_image_from_desktop(img_name)
    thr = threshold_otsu(np.array(img))
    img_bin = binarize_image(img, thr)

    fig, ax = plt.subplots(2, 1)
    ax[0].image(img)
    ax[1].image(img_bin)

