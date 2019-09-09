import numpy as np


def softmax(Z):
    # numerical stability
    A = np.exp(Z - np.max(Z, axis=1, keepdims=True))/np.sum(
        np.exp(Z), axis=1, keepdims=True)
    return A


def SoftmaxLoss(X, labels):
    m = labels.shape[1]
    predictions = softmax(X)
    log_likelihood = -np.log(predictions[range(m), labels])
    loss = np.sum(log_likelihood) / m

    grad = predictions.copy()
    grad[range(m), labels] -= 1
    grad /= m
    return loss, grad
