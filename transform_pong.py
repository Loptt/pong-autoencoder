import sys
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

if len(sys.argv) < 3:
    print("Usage: transform_pong <start> <end>")
    exit()

try:
    start = int(sys.argv[1])
    end = int(sys.argv[2])
except:
    print("Invalid arguments: start, end")

# The following numbers represent the fraction of the image to be cropped in the Y axis.
# These numbers come from the experimental analysis done in image_analysis.ipynb
Y_CROP_START = 34
Y_CROP_END = 191

count = 0

for i in range(end - start + 1):
    try:
        img = load_img(f"images/pong_{start + i}.png")
    except:
        print(f"Unable to open image {start + i}")
        continue
    img_gray = img.convert("L")
    img_arr = img_to_array(img_gray)
    img_cropped = img_arr[Y_CROP_START:Y_CROP_END]
    img = array_to_img(img_cropped)
    img.save(f"images_trans/pong_trans_{start + i}.png")
    count += 1

print(f"Transformed {count} images")
