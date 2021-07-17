from keras.metrics import Metric
import tensorflow as tf
import numpy as np
from .ball_finder import find_balls_tf


def build_coordinate_array(batch_size, screen_width, screen_height):
    result = np.zeros((screen_height, screen_width, 2), np.float32)
    for x in range(screen_width):  # todo: look up numpy mesh to do this faster?
        for y in range(screen_height):
            result[y, x, 0] = x
            result[y, x, 1] = y
    # should be shape [batch_size, screen_height, screen_width, 2]
    result = np.stack([result]*batch_size, axis=0)
    return tf.constant(result)


def build_coordinate_array_tf(batch_size, width, height):
    y, x = tf.meshgrid(range(height), range(width))
    coords = tf.map_fn(lambda a: tf.stack([a[0], a[1]],
                                          axis=1), tf.stack([x, y], axis=1))

    coords = tf.reshape(coords, [1, coords.shape[0], coords.shape[1], 2])
    batch_coords = tf.tile(coords, tf.constant([batch_size, 1, 1, 1]))

    return tf.cast(batch_coords, tf.float32)


def calculate_ball_loss(ball_locations, reconstruction, coordinate_array):
    # on entry, ball locations is shape [batch, 2]  example, [[original_image1_ballx,original_image1_bally],[original_image2_ballx,original_image2_bally],...]
    # reconstraction shape [batch, screen_height, screen_width,1]???
    # coordinate_array is shape [batch_size, screen_height, screen_width, 2]

    coordinate_array_temp = coordinate_array

    batch_size = ball_locations.shape[0]
    screen_width = reconstruction.shape[2]
    screen_height = reconstruction.shape[1]
    try:
        assert coordinate_array.shape[0] == batch_size
    except AssertionError:
        # print(
        #    f"Batch Size {batch_size}, Coord array batch {coordinate_array.shape[0]}")
        coordinate_array_temp = coordinate_array[:batch_size]

    assert coordinate_array_temp.shape[1] == screen_height
    assert coordinate_array_temp.shape[2] == screen_width

    ball_locations_r = tf.reshape(ball_locations, [batch_size, 1, 1, 2])
    relative_locations = coordinate_array_temp - \
        tf.cast(ball_locations_r, tf.float32)
    relative_locations_squared = tf.square(relative_locations)
    # will work out x*x+y*y.  Output shape should be [batch_size, screen_height, screen_widht)
    temp = tf.reduce_sum(relative_locations_squared, axis=3)
    # Output shape should be [batch_size, screen_height, screen_widht)
    pythagorean_distances = tf.sqrt(temp)

    # shape should be [batch_size, screen_height, screen_width)
    reconstructed_image = reconstruction[:, :, :, 0]
    # should leave us with a vector of shape [batch_size]

    # # this will slightly brighten every pixel.  It will ensure
    # a) we can't get a division by zero when normalising the brigtness.
    # b) That the neural network cannot learn to cheat by setting the brigthness of every pixel to zero.
    brightened_reconstructed_image = reconstructed_image+1e-6
    reconstucted_image_brightness = tf.reduce_sum(
        brightened_reconstructed_image, axis=[1, 2])
    normalised_brightness_image = brightened_reconstructed_image / \
        tf.reshape(reconstucted_image_brightness, [batch_size, 1, 1])

    # shape should be [batch_size, screen_height, screen_width)
    score = normalised_brightness_image*pythagorean_distances
    # shape should be [batch_size)
    score = tf.reduce_sum(score, axis=[1, 2])
    result = tf.reduce_mean(score)

    return result


def ball_track_metric(y_true, y_pred):
    #print("SHAPE", tf.shape(y_true).numpy())
    coordinate_array = build_coordinate_array(
        y_true.shape[0], y_true.shape[2], y_true.shape[1])
    ball_locations = find_balls_tf(y_true)

    return calculate_ball_loss(ball_locations, y_pred, coordinate_array)
