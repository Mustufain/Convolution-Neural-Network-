import numpy as np
from utils.cnn_utils import load_dataset
from loss.softmax import softmaxloss
from layers.convolution import Convolution
from layers.relu import Relu
from layers.pooling import Maxpool
from layers.flatten import Flatten
from layers.dense import Dense
from model.cnn import CNN
import copy
from tqdm import tqdm


def make_cnn(input_dim, num_of_classes):
    conv1 = Convolution(input_dim=input_dim, pad=2,
                        stride=2,
                        num_filters=10,
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


def load_data():
    train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
    num_of_classes = len(classes)
    train_set_x = train_set_x / 255
    test_set_x = test_set_x / 255
    return train_set_x, train_set_y, test_set_x, test_set_y, num_of_classes

def make_model(train_set_x, n_class):
    input_dim = train_set_x.shape[1:]
    layers = make_cnn(input_dim, n_class)
    cnn = CNN(layers)
    return cnn

def params_to_vector(params):
    count = 0
    for index, param in enumerate(params):
        if len(param) is not 0:
            w = np.reshape(param[0], (-1, 1))
            b = np.reshape(param[1], (-1, 1))
            new_vector = np.concatenate((w, b), axis=0)
            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1
    return theta

def grads_to_vector(grads):
    count = 0
    grads = list(reversed(grads))
    for index, param in enumerate(grads):
        if len(param) is not 0:
            w = np.reshape(param[0], (-1, 1))
            b = np.reshape(param[1], (-1, 1))
            new_vector = np.concatenate((w, b), axis=0)
            if count == 0:
                theta = new_vector
            else:
                theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1
    return theta

def vector_to_param(theta, params):
    new_param = copy.deepcopy(params)
    start = 0
    for index in range(len(params)):
        if len(params[index]) is not 0: # check for learnable layers
            for i in range(len(params[index])):
                end = start + np.prod(params[index][i].shape)
                p = theta[start:end].reshape(params[index][i].shape)
                new_param[index][i] = p
                start = end
        else:
            new_param[index] = []

    return new_param

def compare(new_param, old_param):
    different_value = []
    for index in range(len(new_param)):
        if len(new_param[index]) is not 0:
            w_value = np.sum(new_param[index][0] != old_param[index][0])
            b_value = np.sum(new_param[index][1] != old_param[index][1])
            different_value.append(w_value)
            different_value.append(b_value)
        else:
            assert (new_param[index] == old_param[index])
    return sum(different_value)

def grad_check():

    train_set_x, train_set_y, test_set_x, test_set_y, n_class = load_data()
    # select randomly 2 data points from training data
    n = 2
    index = np.random.choice(train_set_x.shape[0], n)
    train_set_x = train_set_x[index]
    train_set_y = train_set_y[:, index]
    cnn = make_model(train_set_x, n_class)
    print (cnn.layers)
    A = cnn.forward(train_set_x)
    loss, dA = softmaxloss(A, train_set_y)
    assert (A.shape == dA.shape)
    grads = cnn.backward(dA)
    grads_values = grads_to_vector(grads)
    initial_params = cnn.params
    parameters_values = params_to_vector(initial_params) # initial parameters
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    print ('number of parameters: ', num_parameters)
    epsilon = 1e-7
    assert (len(grads_values) == len(parameters_values))
    for i in tqdm(range(0, num_parameters)):


        thetaplus = copy.deepcopy(parameters_values)
        thetaplus[i][0] = thetaplus[i][0] + epsilon # parameters
        new_param = vector_to_param(thetaplus, initial_params)
        difference = compare(new_param, initial_params)
        assert ( difference == 1) # make sure only one parameter is changed
        cnn.params = new_param
        A = cnn.forward(train_set_x)
        J_plus[i], _ = softmaxloss(A, train_set_y)

        thetaminus = copy.deepcopy(parameters_values)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        new_param = vector_to_param(thetaminus, initial_params)
        difference = compare(new_param, initial_params)
        assert (difference == 1)  # make sure only one parameter is changed
        cnn.params = new_param
        A = cnn.forward(train_set_x)
        J_minus[i], _ = softmaxloss(A, train_set_y)

        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    numerator = np.linalg.norm(gradapprox - grads_values)
    denominator = np.linalg.norm(grads_values) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference > 2e-7:
        print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(
                difference) + "\033[0m")
    else:
        print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(
                difference) + "\033[0m")

    return difference





