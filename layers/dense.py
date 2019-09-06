import numpy as np


class Dense(object):

    def __init__(self, A_prev, output_neurons, layer_type='hidden'):
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
        else:
            return Z

    def relu(self, Z):
        A = np.maximum(0, Z)  # element-wise
        return A

    def relu_backward(self):
        return

    def backward(self):
        return
