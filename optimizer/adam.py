import numpy as np


class Adam(object):

    def __init__(self):
        self.learning_rate = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-08

    def update_parameters(self):
        
