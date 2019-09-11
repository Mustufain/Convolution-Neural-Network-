import numpy as np


class Relu(object):

    def __init__(self):
        self.params = []

    def forward(self, Z):
        """
        Forward propogation of relu layer.

        Parameters:
        Z -- Input data -- numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        A -- Activations of relu layer-- numpy array of shape m, n_H_prev, n_W_prev, n_C_prev)

        """
        self.Z = Z
        A = np.maximum(0, Z)  # element-wise
        return A

    def backward(self, dA):
        """
        Backward propogation of relu layer.

        f′(x) = {1 if x > 0}
                {0 otherwise}

        Parameters:
        dA -- gradient of cost with respect to the output of the relu layer,
              same shape as A

        Returns:
        dZ -- gradient of cost with respect to the input of the relu layer,
              same shape as Z

        """
        Z = self.Z
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert (dZ.shape == self.Z.shape)
        return dZ, []
