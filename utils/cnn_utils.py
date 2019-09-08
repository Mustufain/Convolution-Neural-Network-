import numpy as np
import h5py


def load_dataset():
    """
    Load the dataset
    """
    train_dataset = h5py.File('dataset/train_signs.h5', "r")
    train_set_x = np.array(train_dataset["train_set_x"][:])
    train_set_y = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('dataset/test_signs.h5', "r")
    test_set_x = np.array(test_dataset["test_set_x"][:])
    test_set_y = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y_orig = test_set_y.reshape((1, test_set_y.shape[0]))

    return train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, classes


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
