import numpy as np


class Relu(object):

    def __init__(self):
        self.params = []

    def forward(self, Z):
        self.Z = Z
        A = np.maximum(0, Z)  # element-wise
        return A

    def backward(self, dA):
        """
        f′(x) = { 1 if x>0   }
                { 0 otherwise}
        """
        Z = self.Z
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert (dZ.shape == self.Z.shape)
        return dZ, []
