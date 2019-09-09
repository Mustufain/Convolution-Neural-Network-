import numpy as np
from utils.cnn_utils import get_minibatches
from tqdm import tqdm


class Adam(object):

    def __init__(self, model, X_train, y_train,
                 learning_rate, epoch, minibatch_size, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-08
        self.epoch = epoch
        self.X_test = X_test
        self.y_test = y_test
        self.minibatch_size = minibatch_size

    def minimize(self):
        minibatches = get_minibatches(self.X_train,
                                      self.y_train,
                                      self.minibatch_size)
        # forward pass
        for i in tqdm(range(self.epoch)):
            velocity, cache = [], []
            for param_layer in self.model.params:
                p = [np.zeros_like(param) for param in list(param_layer)]
                velocity.append(p)
                cache.append(p)
        # weight update
        t = 1
        for X_mini, y_mini in minibatches:
            loss, grads = self.model.fit(X_mini, y_mini)
            for c, v, param, grad, in zip(cache, velocity, self.model.params, reversed(grads)):
                for i in range(len(grad)):
                    c[i] = self.beta1 * c[i] + (1. - self.beta1) * grad[i]
                    v[i] = self.beta2 * v[i] + (1. - self.beta2) * (grad[i]**2)
                    mt = c[i] / (1. - self.beta1**(t))
                    vt = v[i] / (1. - self.beta2**(t))
                    param[i] += - self.learning_rate * mt / (np.sqrt(vt) + 1e-4)
            t += 1
