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

    return tf.constant(dataset, dtype=tf.float64), tf.constant(validation, dtype=tf.float64)
