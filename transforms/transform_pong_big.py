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
X_CROP_START = 2
X_CROP_END = 159
X_CROP_START_RESIZE = 5
X_CROP_END_RESIZE = 34
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

    input_img_arr = np.reshape(
        input_img_arr, (1, *img_array.shape))

    rescaled = scaler.predict(input_img_arr)
    rescaled = np.reshape(
        rescaled, (rescaled.shape[1], rescaled.shape[2], rescaled.shape[3]))

    return rescaled


def dilate_img(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dilated = cv2.dilate(img, kernel, iterations=1)
    return np.reshape(dilated, img.shape)


def combine_imgs(img_paddles, img_ball):
    print("BALL SHAPE", img_ball.shape)
    img_combined = np.copy(img_paddles)
    img_combined[:, X_CROP_START_RESIZE:X_CROP_END_RESIZE, :] = img_ball
    return img_combined


def conversion_pipe(pil_image):
    img_gray = pil_image.convert("L")
    img_arr = img_to_array(img_gray)
    img_cropped = img_arr[Y_CROP_START:Y_CROP_END, X_CROP_START: X_CROP_END]
    img_mono = gray_to_mono(img_cropped)
    # Downscale produces one more pixel in the right side, breaking symmetry
    # So we remove it
    img_resized = maxpool_downscale(img_mono)[:, :-1]
    # Not the case in ball down scale
    img_ball = img_resized[:, X_CROP_START_RESIZE:X_CROP_END_RESIZE]
    img_dilated = dilate_img(img_ball)
    img_combined = combine_imgs(img_resized, img_dilated)
    img = array_to_img(img_combined)

    return img


for i in range(end - start + 1):
    try:
        img = load_img(f"../images/pong_{start + i}.png")
    except:
        print(f"Unable to open image {start + i}")
        continue
    img_converted = conversion_pipe(img)
    img_converted.save(
        f"../images_big/pong_big_{start + i}.png")
    print("Processed", start + i)
    count += 1

print(f"Transformed {count} images")
