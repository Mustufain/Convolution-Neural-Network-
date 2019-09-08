import numpy as np


class Dense(object):

    def __init__(self, A_prev, output_neurons, layer_type='hidden'):
        np.random.seed(1)
        self.layer_type = layer_type
        self.A_prev = A_prev
        self.input_neurons = np.prod(self.A_prev.shape)
        self.output_neurons = output_neurons
        self.W = np.random.normal(  # Xavier Initialization
            loc=0.0, scale=np.sqrt(
                2 / ((self.input_neurons))),
            size=(self.input_neurons, self.output_neurons))
        self.b = np.zeros(shape=(self.output_neurons))

    def forward(self):
        flatten = self.A_prev.flatten()
        flatten = flatten.reshape(flatten.shape[0], 1).T
        Z = np.dot(flatten, self.W) + self.b
        if self.layer_type == 'hidden':
            A = self.relu(Z)
            return A
        else: # last layer
            self.Z = Z
            return Z

    def relu(self, Z):
        A = np.maximum(0, Z)  # element-wise
        return A

    def relu_backward(self, dA):
        """
        f′(x) = { 1 if x>0   }
                { 0 otherwise}
        """
        Z = self.Z
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def backward_dense(self, dZ):
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

    def backward(self, dA):
        """
        Implement the backward propagation
        for the FC->RELU layer.
        """
        dZ = self.relu_backward(dA)
        dA_prev, dW, db = self.backward_dense(dZ)
        return dA_prev, dW, db
