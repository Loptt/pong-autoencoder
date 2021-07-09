from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from ball_finder import find_balls, find_balls_tf


img = load_img('../images_trans/pong_trans_5000.png', color_mode='grayscale')
img_arr = img_to_array(img) / 255

balls = find_balls(img_arr, 5, 34)

print(balls)
