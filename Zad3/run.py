import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import gzip
import cv2
from numpy.lib.function_base import average
from mlp import MLP

image_size = 28
num_images = 5


def read_from_idx_files(file_name):
    arr = idx2numpy.convert_from_file(file_name)
    return arr


data_file_names = {
    "train_data": "data/train-images.idx3-ubyte",
    "train_labels": "data/train-labels.idx1-ubyte",
    "test_images": "data/t10k-images.idx3-ubyte",
    "test_labels": "data/t10k-labels.idx1-ubyte"
}

if __name__ == "__main__":
    TRAINING_DATA = read_from_idx_files(data_file_names["train_data"])
    TRAINING_LABELS = read_from_idx_files(data_file_names["train_labels"])
    vector_shape = TRAINING_DATA.shape
    starting_neurons = vector_shape[1] * vector_shape[2]
    LAYERS = 4
    LABELS = 10
    bias = []
    learning_factors = []
    mlp = MLP(starting_neurons=starting_neurons, layers=LAYERS, labels=LABELS,
              bias=bias, learning_factor=learning_factors)
    mlp.train(TRAINING_DATA, TRAINING_LABELS)
