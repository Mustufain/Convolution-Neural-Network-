
class Flatten(object):

    def __init__(self):
        self.params = []

    def forward(self, X):
        self.X_shape = X.shape
        self.out_shape = (self.X_shape[0], -1)
        Z = X.ravel().reshape(self.out_shape)
        self.out_shape = self.out_shape[1]
        return Z

    def backward(self, dA):
        dA_prev = dA.reshape(self.X_shape)
        assert (dA_prev.shape == self.out.shape)
        return dA_prev, ()
