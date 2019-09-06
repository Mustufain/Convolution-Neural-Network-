import numpy as np


class Softmax(object):

    def __init__(self, A_prev, output_neurons):
        self.A_prev = A_prev
        self.input_neurons = A_prev.shape[1]
        self.output_neurons = output_neurons
        self.W = np.random.normal(  # Xavier Initialization
            loc=0.0, scale=np.sqrt(
                2 / ((self.input_neurons))),
            size=(self.input_neurons, self.output_neurons))
        self.b = np.zeros(shape=(self.output_neurons))

    def forward(self):
        Z = np.dot(self.A_prev, self.W) + self.b
        A = self.softmax(Z)
        return A

    def softmax(self, Z):
        A = np.exp(Z)/sum(np.exp(Z))
        return A

    def backward(self):
        return
