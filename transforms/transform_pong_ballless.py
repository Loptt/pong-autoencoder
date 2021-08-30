import sys

import os
os.add_dll_directory(
    "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.4\\bin")

import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from skimage.measure import block_reduce

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# The following numbers represent the fraction of the image to be cropped in the Y and X axis.
# These numbers come from the experimental analysis done in image_analysis.ipynb
Y_CROP_START = 34
Y_CROP_END = 191
X_CROP_START = 1
X_CROP_END = 161
RESIZE_RATIO = 0.5

count = 0

if len(tf.config.list_physical_devices('GPU')) < 1:
    print("GPU cannot be configured")
    exit()
else:
    print("GPU configured properly")


def gray_to_mono(img_array):
    img_mono = np.copy(img_array)

    for i, x in enumerate(img_mono):
        for j, y in enumerate(x):
            for k, v in enumerate(y):
                if v > 100:
                    img_mono[i, j, k] = 255
                else:
                    img_mono[i, j, k] = 0

    return img_mono


def gray_to_mono_fast(img_array):
    img_tf = tf.constant(img_array)

    # Experimental analysis reveals the mininum value in the screen is 87.0
    # So subtract this value to set the minimum to 0.
    img_tf = img_tf - 87.0

    img_tf = img_tf / 255.0
    img_tf = tf.math.ceil(img_tf)
    img_tf = img_tf * 255.0

    return img_tf.numpy()


def block_reduce_downscale(img):
    img_r = np.reshape(img, img.shape[:-1])
    reduced = block_reduce(img_r, (8, 8), np.max)
    return np.reshape(reduced, (*reduced.shape, 1))


def tf_nn_downscale(img):
    img_tf = tf.reshape(tf.constant(img), (1, img.shape[0], img.shape[1], 1))
    # reduced = block_reduce(img_r, (4, 4), np.max)
    reduced = tf.nn.max_pool(img_tf, ksize=8, strides=8, padding='SAME')
    reduced = tf.reshape(reduced, reduced.shape[1:])
    return reduced.numpy()


def dilate_img(img, amount):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (amount, amount))
    dilated = cv2.dilate(img, kernel, iterations=1)
    return np.reshape(dilated, img.shape)


def erode_img(img, amount):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (amount, amount))
    dilated = cv2.erode(img, kernel, iterations=1)
    return np.reshape(dilated, img.shape)


def conversion_pipe_slow(pil_image):
    img_gray = pil_image.convert("L")
    img_arr = img_to_array(img_gray)
    img_cropped = img_arr[Y_CROP_START:Y_CROP_END, X_CROP_START:X_CROP_END]
    img_mono = gray_to_mono(img_cropped)
    img_eroded = erode_img(img_mono, 4)
    img_dilated = dilate_img(img_eroded, 4)
    img_resized = block_reduce_downscale(img_dilated)
    img_dilated = dilate_img(img_resized, 2)
    img = array_to_img(img_dilated)
    # img = img.resize(
    #    (int(img.width * RESIZE_RATIO), int(img.height * RESIZE_RATIO)))

    return img


def conversion_pipe(pil_image):
    img_gray = pil_image.convert("L")
    img_arr = img_to_array(img_gray)
    img_cropped = img_arr[Y_CROP_START:Y_CROP_END, X_CROP_START:X_CROP_END]
    img_mono = gray_to_mono_fast(img_cropped)
    img_eroded = erode_img(img_mono, 4)
    img_dilated = dilate_img(img_eroded, 4)
    img_resized = tf_nn_downscale(img_dilated)
    img_dilated = dilate_img(img_resized, 2)
    img = array_to_img(img_dilated)
    # img = img.resize(
    #    (int(img.width * RESIZE_RATIO), int(img.height * RESIZE_RATIO)))

    return img


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: transform_pong <start> <end>")
        exit()

    try:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
    except:
        print("Invalid arguments: start, end")

    for i in range(end - start + 1):
        try:
            img = load_img(f"images/pong_{start + i}.png")
        except:
            print(f"Unable to open image {start + i}")
            continue
        img_converted = conversion_pipe(img)
        img_converted.save(
            f"images_ballless/pong_ballless_{start + i}.png")

        print("Processed", start + i)
        count += 1

    print(f"Transformed {count} images")
