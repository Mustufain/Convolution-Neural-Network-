import numpy as np


class Softmax(object):

    def __init__(self, Z_prev, output_neurons):
        np.random.seed(1)
        self.Z_prev = Z_prev
        self.input_neurons = Z_prev.shape[1]
        self.output_neurons = output_neurons
        self.W = np.random.normal(  # Xavier Initialization
            loc=0.0, scale=np.sqrt(
                2 / ((self.input_neurons))),
            size=(self.input_neurons, self.output_neurons))
        self.b = np.zeros(shape=(self.output_neurons))

    def forward(self):
        Z = np.dot(self.Z_prev, self.W) + self.b
        A = self.softmax(Z)
        return A

    def softmax(self, Z):
        A = np.exp(Z)/np.sum(np.exp(Z), axis=1, keepdims=True)
        return A

    def cross_entropy(self, predictions, labels):

        loss = -np.sum((np.log(predictions) * labels), axis=1)
        return loss

    def compute_cost(self, predictions, labels):
        """
        a.
        """
        m = self.Z_prev.shape[0]
        loss = self.cross_entropy(predictions, labels)
        cost = np.squeeze(loss/m)
        return cost

    def conver_prob_into_class(self, predictions):
        highest_prob = np.max(predictions)
        probs_ = np.copy(predictions)
        probs_[probs_ == highest_prob] = 1
        probs_[probs_ < highest_prob] = 0
        return probs_

    def compute_accuracy(self, predictions, labels):
        predicted_labels = self.conver_prob_into_class(predictions)
        return (predicted_labels == labels).all(axis=0).mean()

    def backward_cross_entropy(self, predictions, labels):

        return predictions - labels

    def backward(self, predictions, labels):
        return predictions - labels
