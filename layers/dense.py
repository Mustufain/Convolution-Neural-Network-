import numpy as np


class Dense(object):

    def __init__(self, input_dim, output_dim, seed):
        self.seed = seed
        np.random.seed(self.seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(
            self.input_dim, self.output_dim) * np.sqrt(2 / np.prod(self.input_dim) + np.prod(self.output_dim))
        #self.W = np.random.normal(  # Xavier Initialization
        #    loc=0.0, scale=np.sqrt(
        #        2 / ((self.input_dim))),
        #    size=(self.input_dim, self.output_dim))
        self.b = np.zeros(shape=(self.output_dim))
        self.params = [self.W, self.b]

    def forward(self, A_prev):
        """
        Forward propogation of Dense layer.

        Parameters:
        A_prev -- input data -- numpy of array shape (m, input_dim)

        Returns:
        Z -- flatten numpy array of shape (m, output_dim)

        """
        np.random.seed(self.seed)
        m = A_prev.shape[0]
        self.A_prev = A_prev
        Z = np.dot(self.A_prev, self.params[0]) + self.params[1]
        assert (Z.shape == (m, self.output_dim))
        return Z

    def backward(self, dA):
        """
        Backward propogation for Dense layer.

        Parameters:
        dA -- gradient of cost with respect to the output of the Dense layer,
              same shape as Z

        Returns:
        dA_prev -- gradient of cost with respect to the input of the Dense layer,
                   same shape as A_prev

        """
        np.random.seed(self.seed)
        m = self.A_prev.shape[0]
        dW = 1./m * np.dot(self.A_prev.T, dA)
        db = 1./m * np.sum(dA, axis=0)
        dA_prev = np.dot(dA, self.W.T)
        assert (dA_prev.shape == self.A_prev.shape)
        assert (dW.shape == self.params[0].shape)
        assert (db.shape == self.params[1].shape)

        return dA_prev, [dW, db]
