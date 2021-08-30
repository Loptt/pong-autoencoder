import sys
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.layers import Input, MaxPool2D
from keras import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if len(sys.argv) < 3:
    print("Usage: transform_pong <start> <end>")
    exit()

try:
    start = int(sys.argv[1])
    end = int(sys.argv[2])
except:
    print("Invalid arguments: start, end")

if len(tf.config.list_physical_devices('GPU')) < 1:
    print("GPU cannot be configured")
    exit()
else:
    print("GPU configured properly")


# The following numbers represent the fraction of the image to be cropped in the Y and X axis.
# These numbers come from the experimental analysis done in image_analysis.ipynb
Y_CROP_START = 34
Y_CROP_END = 191
X_CROP_START = 2
X_CROP_END = 159
RESIZE_RATIO = 0.5

count = 0


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


def maxpool_downscale(img_array):
    input_img_arr = np.copy(img_array)
    input_img = Input(shape=img_array.shape)
    x = MaxPool2D((2, 2), padding='same')(input_img)
    output = MaxPool2D((2, 2), padding='same')(x)

    scaler = Model(input_img, output)
    scaler.compile()

    orig_shape = img_array.shape

    input_img_arr = np.reshape(
        input_img_arr, (1, orig_shape[0], orig_shape[1], orig_shape[2]))

    rescaled = scaler.predict(input_img_arr)
    rescaled = np.reshape(
        rescaled, (rescaled.shape[1], rescaled.shape[2], rescaled.shape[3]))

    return rescaled


def conversion_pipe_slow(pil_image):
    img_gray = pil_image.convert("L")
    img_arr = img_to_array(img_gray)
    img_cropped = img_arr[Y_CROP_START:Y_CROP_END]
    img_cropped = img_cropped[:, X_CROP_START:X_CROP_END]
    img_mono = gray_to_mono(img_cropped)
    #img_resized = maxpool_downscale(img_mono)
    img = array_to_img(img_mono)
    # img = img.resize(
    #    (int(img.width * RESIZE_RATIO), int(img.height * RESIZE_RATIO)))

    return img


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
    reduced = tf.nn.max_pool(img_tf, ksize=4, strides=4, padding='SAME')
    reduced = tf.reshape(reduced, reduced.shape[1:])
    return reduced.numpy()


def conversion_pipe(pil_image):
    img_gray = pil_image.convert("L")
    img_arr = img_to_array(img_gray)
    img_cropped = img_arr[Y_CROP_START:Y_CROP_END, X_CROP_START:X_CROP_END]
    img_mono = gray_to_mono_fast(img_cropped)
    img_resized = tf_nn_downscale(img_mono)
    img = array_to_img(img_resized)
    # img = img.resize(
    #    (int(img.width * RESIZE_RATIO), int(img.height * RESIZE_RATIO)))

    return img


for i in range(end - start + 1):
    try:
        img = load_img(f"images/pong_{start + i}.png")
    except:
        print(f"Unable to open image {start + i}")
        continue
    img_converted = conversion_pipe(img)
    img_converted.save(f"images_trans/pong_trans_{start + i}.png")
    print("Processed", start + i)
    count += 1

print(f"Transformed {count} images")
