import numpy as np


class Flatten(object):

    def __init__(self):
        self.params = []

    def forward(self, A_prev):
        """
        Forward propogation of flatten layer.

        Parameters:
        A_prev -- input data -- numpy of array shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        Z -- flatten numpy array of shape (m, n_H_prev * n_W_prev * n_C_prev)

        """
        self.A_prev = A_prev
        output = np.prod(self.A_prev.shape[1:])
        m = self.A_prev.shape[0]
        self.out_shape = (self.A_prev.shape[0], -1)
        Z = self.A_prev.ravel().reshape(self.out_shape)
        assert (Z.shape == (m, output))
        return Z

    def backward(self, dA):
        """
        Backward propogation of flatten layer.

        Parameters:
        dA -- gradient of cost with respect to the output of the flatten layer,
              same shape as Z

        Returns:
        dA_prev -- gradient of cost with respect to the input of the flatten layer,
                   same shape as A_prev

        """
        dA_prev = dA.reshape(self.A_prev.shape)
        assert (dA_prev.shape == self.A_prev.shape)
        return dA_prev, []
