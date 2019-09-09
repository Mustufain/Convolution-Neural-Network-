import numpy as np


class Dense(object):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.normal(  # Xavier Initialization
            loc=0.0, scale=np.sqrt(
                2 / ((self.input_dim))),
            size=(self.input_dim, self.output_dim))
        self.b = np.zeros(shape=(self.output_dim))
        self.params = [self.W, self.b]

    def forward(self, A_prev):
        m = A_prev.shape[0]
        self.A_prev = A_prev
        Z = np.dot(self.A_prev, self.W) + self.b
        assert (Z.shape == (m, self.output_dim))
        return Z

    def backward(self, dZ):
        """
        Implement the backward propogation
        for FC layer.
        """
        m = self.A_prev.shape[0]
        dW = 1/m * np.dot(dZ, np.transpose(self.A_prev))
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(np.transpose(self.W), dZ)

        assert (dA_prev.shape == self.A_prev.shape)
        assert (dW.shape == self.W.shape)
        assert (db.shape == self.b.shape)

        return dA_prev, dW, db
