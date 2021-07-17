import tensorflow as tf
import random


# Generates a dataset consisting of nxn squares that have a single
# point with value 1, and the rest with value 0. Every single possible
# data point is generated, i.e. with the 1 value in every possible square
# so the size of the dataset will depend on the initial size given

def print_matrix(m):
    print(" ")
    for i in range(len(m)):
        for j in range(len(m[0])):
            print(m[i][j], end=' ')

        print(" ")

    print(" ")


def generate_matrix(n, m):
    matrix = []

    for i in range(n):
        row = []
        for j in range(m):
            row.append([0.0])
        matrix.append(row)

    return matrix


def get_coords(data):
    # Assuming data is tensor shape (batch_size, height, width , 1)

    y = tf.math.reduce_max(tf.math.argmax(data, axis=1), axis=1)
    x = tf.math.reduce_max(tf.math.argmax(data, axis=2), axis=1)
    return tf.concat([x, y], axis=1)

# Generates nxm matrix dataset of zeroes with a single 1, in every position possible


def generate_single_point(n, m, validation_split=0.2):
    dataset = []

    for i in range(n):
        for j in range(m):
            new_matrix = generate_matrix(n, m)
            new_matrix[i][j][0] = 1.0
            dataset.append(new_matrix)

    val_amount = int(len(dataset) * validation_split)
    validation = []

    val_idcs = random.sample(range(0, len(dataset)), val_amount)
    val_idcs.sort()

    random.shuffle(dataset)

    for i in val_idcs:
        validation.append(dataset[i])

    for i in reversed(val_idcs):
        dataset.pop(i)

    return tf.constant(dataset, dtype=tf.float32), tf.constant(validation, dtype=tf.float32)

# Generates nxm matrix dataset of zeroes with a single 1, in every position possible, it also
# returns the coords for each matrix


def generate_single_point_coords(n, m, validation_split=0.2):
    dataset = []
    coords = []

    for i in range(n):
        for j in range(m):
            new_matrix = generate_matrix(n, m)
            new_matrix[i][j][0] = 1.0
            dataset.append(new_matrix)

    val_amount = int(len(dataset) * validation_split)
    validation = []

    val_idcs = random.sample(range(0, len(dataset)), val_amount)
    val_idcs.sort()

    random.shuffle(dataset)

    for i in val_idcs:
        validation.append(dataset[i])

    for i in reversed(val_idcs):
        dataset.pop(i)

    dataset_tf = tf.constant(dataset, dtype=tf.float32)
    validation_tf = tf.constant(validation, dtype=tf.float32)

    dataset_coords = get_coords(dataset_tf)
    validation_coords = get_coords(validation_tf)

    return dataset_tf, dataset_coords, validation_tf, validation_coords


def generate_matrix_from_coords(n, m, coords):
    mat = generate_matrix(n, m)

    x = int(coords[0])
    y = int(coords[1])

    mat[y][x] = [1.0]

    return tf.constant(mat, dtype=tf.float32)
