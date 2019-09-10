import numpy as np


class Flatten(object):

    def __init__(self):
        self.params = []

    def forward(self, A_prev):
        self.A_prev = A_prev
        output = np.prod(self.A_prev.shape[1:])
        m = self.A_prev.shape[0]
        self.out_shape = (self.A_prev.shape[0], -1)
        Z = self.A_prev.ravel().reshape(self.out_shape)
        assert (Z.shape == (m, output))
        return Z

    def backward(self, dA):
        dA_prev = dA.reshape(self.A_prev.shape)
        assert (dA_prev.shape == self.A_prev.shape)
        return dA_prev, []
