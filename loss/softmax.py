import numpy as np
from utils.cnn_utils import convert_to_one_hot

def softmax(Z):
    # numerical stability
    Z = Z - np.expand_dims(np.max(Z, axis = 1), 1)
    Z = np.exp(Z)
    ax_sum = np.expand_dims(np.sum(Z, axis=1), 1)

    # finally: divide elementwise
    A = Z / ax_sum
    return A


def SoftmaxLoss(X, labels):
    one_hot_labels = convert_to_one_hot(labels, 6)
    m = labels.shape[0]
    predictions = softmax(X)
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    loss = -np.sum(one_hot_labels * np.log(predictions + 1e-9)) / N
    grad = predictions.copy()
    grad[range(N), labels] -= 1
    grad /= N
    return loss, grad


    #log_likelihood = -np.log(predictions[range(m), one_hot_labels])
    #loss = np.sum(log_likelihood) / m
    #grad = predictions.copy()
    #grad[range(m), labels] -= 1
    #grad /= m
    #return loss, grad
