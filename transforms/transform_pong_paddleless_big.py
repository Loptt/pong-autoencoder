import sys
import os
import numpy as np
import cv2
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

# The following numbers represent the fraction of the image to be cropped in the Y and X axis.
# These numbers come from the experimental analysis done in image_analysis.ipynb
Y_CROP_START = 34
Y_CROP_END = 191
X_CROP_START = 20
X_CROP_END = 140
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


def check_ball(img_mono):
    for i, x in enumerate(img_mono):
        for j, y in enumerate(x):
            for k, v in enumerate(y):
                if v >= 255:
                    return True

    return False


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


def dilate_img(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dilated = cv2.dilate(img, kernel, iterations=1)
    return np.reshape(dilated, img.shape)


def conversion_pipe(pil_image):
    img_gray = pil_image.convert("L")
    img_arr = img_to_array(img_gray)
    img_cropped = img_arr[Y_CROP_START:Y_CROP_END]
    img_cropped = img_cropped[:, X_CROP_START:X_CROP_END]
    img_mono = gray_to_mono(img_cropped)
    has_ball = check_ball(img_mono)
    if not has_ball:
        return img_mono, False
    img_resized = maxpool_downscale(img_mono)
    img_dilated = dilate_img(img_resized)
    img = array_to_img(img_dilated)

    return img, True


for i in range(end - start + 1):
    try:
        img = load_img(f"images/pong_{start + i}.png")
    except:
        print(f"Unable to open image {start + i}")
        continue
    img_converted, has_ball = conversion_pipe(img)
    if has_ball:
        img_converted.save(
            f"images_paddleless_big/pong_paddleless_big_{start + i}.png")
    else:
        print(f"Image {start + i} contains no ball, not saved.")
    print("Processed", start + i)
    count += 1

print(f"Transformed {count} images")
