import numpy as np
import h5py
import math


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
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def conver_prob_into_class(predictions):
    highest_prob = np.max(predictions)
    probs_ = np.copy(predictions)
    probs_[probs_ == highest_prob] = 1
    probs_[probs_ < highest_prob] = 0
    return probs_


def accuracy(predictions, labels):
    labels = labels.T
    assert (predictions.shape == labels.shape)
    return np.mean(predictions == labels)


def get_minibatches(X, Y, mini_batch_size=64):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (number of examples, n_width, n_height, n_channel)
    Y -- true "label" vector, of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(1)
    m = X.shape[0]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[
            mini_batch_size * k:mini_batch_size * (k+1), :, :, :]
        mini_batch_Y = shuffled_Y[
            :, mini_batch_size * k:mini_batch_size * (k+1)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[mini_batch_size*(k+1):m, :, :, :]
        mini_batch_Y = shuffled_Y[:, mini_batch_size*(k+1):m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
