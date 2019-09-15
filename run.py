from model.cnn import CNN
from layers.convolution import Convolution
from layers.relu import Relu
from layers.pooling import Maxpool
from layers.dense import Dense
from layers.flatten import Flatten
from optimizer.adam import Adam
from utils.cnn_utils import load_dataset
from gradient_check.gradient_check import grad_check
import numpy as np
import pickle
import sys

def make_cnn(input_dim, num_of_classes):
    conv1 = Convolution(input_dim=input_dim, pad=2,
                        stride=2,
                        num_filters=1,
                        filter_size=3, seed=1)
    relu1 = Relu()
    maxpool1 = Maxpool(input_dim=conv1.output_dim,
                       filter_size=2,
                       stride=1)
    flatten = Flatten(seed=1)
    dense1 = Dense(input_dim=np.prod(maxpool1.output_dim),
                   output_dim=num_of_classes, seed=1)

    layers = [conv1, relu1, maxpool1, flatten, dense1]
    return layers


if __name__ == '__main__':

    if len(sys.argv) == 2:
        action = sys.argv[1]
        if action == '--debug':
            # run gradient check on one data point.
            grad_check()
    else:
        train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
        num_of_classes = len(classes)
        train_set_x = train_set_x / 255
        test_set_x = test_set_x / 255

        input_dim = train_set_x.shape[1:]
        layers = make_cnn(input_dim, num_of_classes)
        cnn = CNN(layers)
        cnn, costs = Adam(model=cnn, X_train=train_set_x,
                      y_train=train_set_y, epoch=30,
                      learning_rate=0.001, X_test=test_set_x,
                      y_test=test_set_y, minibatch_size=64).minimize()

        with open("costs.txt", "wb") as fp:  # Pickling
            pickle.dump(costs, fp)
