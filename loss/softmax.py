import numpy as np
from utils.cnn_utils import convert_to_one_hot


def softmax(z):
    """

    :param Z: output of previous layer of shape (m, 6)
    :return: probabilties of shape (m, 6)
    """

    # numerical stability
    z = z - np.expand_dims(np.max(z, axis=1), 1)
    z = np.exp(z)
    ax_sum = np.expand_dims(np.sum(z, axis=1), 1)

    # finally: divide elementwise
    A = z / ax_sum
    return A


def softmaxloss(x, labels):
    """

    :param x: output of previous layer of shape (m, 6)
    :param labels: class labels of shape (1, m)
    :return:
    """

    one_hot_labels = convert_to_one_hot(labels, 6)
    predictions = softmax(x)
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    #loss = -np.sum(one_hot_labels * np.log(predictions + 1e-9)) / N
    loss = -np.sum(one_hot_labels * np.log(predictions)) / N
    grad = predictions.copy()
    grad[range(N), labels] -= 1
    grad /= N
    return loss, grad


