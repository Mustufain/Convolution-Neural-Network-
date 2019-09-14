from loss.softmax import SoftmaxLoss, softmax
import numpy as np


class CNN(object):

    def __init__(self, layers):
        self.layers = layers
        self.params = []
        for layer in self.layers:
            self.params.append(layer.params)

    def forward(self, X):
        for layer in range(0, len(self.layers)):
            # output of last layer serves as input of next layer
            self.layers[layer].params = self.params[layer]
            X = self.layers[layer].forward(X)
        return X

    def backward(self, dA):
        grads = []
        # go in reverse order
        for layer in reversed(self.layers):
            dA, grad = layer.backward(dA)
            grads.append(grad)  # dW, db
        return grads

    def fit(self, X, y):
        """
        Fit the training data.

        Parameters:
        X -- input data, numpy array of shape (m, n_H, n_W, n_C)
        Y -- true "label" vector, numpy array of shape (1, m)

        Returns:
        loss -- Softmax loss -- float
        grads -- gradients backpropogated through each layer

        """
        A = self.forward(X)
        loss, dA = SoftmaxLoss(A, y)
        assert (A.shape == dA.shape)
        grads = self.backward(dA)
        return loss, grads

    def predict(self, X):
        """
        Predict on tests data.

        Parameters:
        X -- input data, numpy array of shape (m, n_H, n_W, n_C)

        Returns:
        prediction -- predictions on tests data, numpy array of shape (m, 1)
        """

        X = self.forward(X)
        prediction = np.argmax(softmax(X), axis=1)
        prediction = prediction.reshape(prediction.shape[0], 1)
        return prediction
