import queue
import numpy as np
import tensorflow as tf


# Assuming we get a n*m numpy array
def find_balls(img, x_start, x_end):
    if not isinstance(img, np.ndarray):
        raise Exception(f"Image must be numpy array, got {type(img)}")

    white_val = np.amax(img)

    # Create a visited matrix with the same shape except the last dim which is the color channel
    visited = np.full(img.shape[:-1], False)
    current_ball = []
    found_balls = []
    q = queue.Queue()
    size = 0

    for i in range(len(img)):
        for j in range(x_start, x_end):
            # If the pixel is not white, ignore
            if img[i, j, 0] < white_val or visited[i, j]:
                continue
            # For each white pixel perfom BFS and calculate size of cluster
            size = 0
            current_ball = []
            q.put((i, j))
            while not q.empty():
                size += 1
                y, x = q.get()
                # Check if the next pixel is in the search range, it has not been visited, and it is a white pixel
                if (x < x_end and x >= x_start and y < len(img) and y >= 0) and not visited[y, x] and img[y, x, 0] >= white_val:
                    visited[y, x] = True
                    current_ball.append((y, x))
                    q.put((y+1, x))
                    q.put((y-1, x))
                    q.put((y, x+1))
                    q.put((y, x-1))

            # After finishing processing a ball, calculate its center by averaging the coordinates
            y_avg = sum([y for y, x in current_ball]) / len(current_ball)
            x_avg = sum([x for y, x in current_ball]) / len(current_ball)

            found_balls.append([x_avg, y_avg])

    found_balls = np.array(found_balls)
    return found_balls


def find_balls_tf(data):
    # Assuming data is tensor shape (batch_size, height, width , 1)

    y = tf.math.reduce_max(tf.math.argmax(data, axis=1), axis=1)
    x = tf.math.reduce_max(tf.math.argmax(data, axis=2), axis=1)
    return tf.concat([x, y], axis=1)
