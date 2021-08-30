import os
os.add_dll_directory(
    "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.4\\bin")

import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle

def load_images(path):
    files = os.listdir(path)
    images = []

    for f in files:
        print("Loading " + path + f)
        img = load_img(path + f, color_mode='grayscale')
        img_arr = img_to_array(img)
        images.append(img_arr)
    
    return tf.constant(images)

def find_balls(data):
    # Assuming data is tensor shape (batch_size, height, width , 1)
    y = tf.math.reduce_max(tf.math.argmax(data, axis=1), axis=1)
    x = tf.math.reduce_max(tf.math.argmax(data, axis=2), axis=1)
    return tf.concat([x, y], axis=1)

def find_paddles(data):
    # Assuming data is tensor shape (batch_size, height, width , 1)
    left_paddle = data[:, :, :10]
    right_paddle = data[:, :, 10:]

    left_y = tf.math.reduce_max(tf.math.argmax(left_paddle, axis=1), axis=1)
    left_x = tf.math.reduce_max(tf.math.argmax(left_paddle, axis=2), axis=1)

    right_y = tf.math.reduce_max(tf.math.argmax(right_paddle, axis=1), axis=1)
    right_x = tf.math.reduce_max(tf.math.argmax(right_paddle, axis=2), axis=1)

    return tf.concat([left_x, left_y, right_x, right_y], axis=1)

if __name__ == '__main__':
    imgs_paddle = load_images("./images_ballless/")
    imgs_ball = load_images("./images_paddleless_big/")

    model_paddle = load_model("./prod_models/vae_ballless")
    model_ball = load_model("./prod_models/vae_big_paddleless")

    latents_paddle = tf.constant(model_paddle.encoder.predict(imgs_paddle)[2])
    latents_ball = tf.constant(model_ball.encoder.predict(imgs_ball)[2])

    paddles_loc = find_paddles(imgs_paddle)
    balls_loc = find_balls(imgs_ball)

    pickle.dump((latents_paddle, paddles_loc), open('./latent_tests/paddle_latent_ds.p', 'wb'))
    pickle.dump((latents_ball, balls_loc), open('./latent_tests/balls_latent_ds.p', 'wb'))

    
