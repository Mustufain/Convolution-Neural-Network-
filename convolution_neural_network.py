from layers.convolution import Convolution
from layers.pooling import Pooling
from layers.dense import Dense
from layers.softmax_layer import Softmax
import numpy as np
import h5py


"""
conv -> relu - > pool -> conv -> relu -> pool -> fc -> relu -> fc -> softmax
"""


def load_dataset():
    """
    Load the dataset
    """
    train_dataset = h5py.File('dataset/train_signs.h5', "r")
    train_set_x = np.array(train_dataset["train_set_x"][:])
    train_set_y = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('dataset/test_signs.h5', "r")
    test_set_x = np.array(test_dataset["test_set_x"][:])
    test_set_y = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y.reshape((1, train_set_y.shape[0]))
    test_set_y_orig = test_set_y.reshape((1, test_set_y.shape[0]))

    return train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, classes


if __name__ == '__main__':
    train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
    train_set_x = train_set_x / 255
    print (classes)
    test_image = train_set_x[0]
    label = train_set_y[0][0]
    print (label)
    conv = Convolution(test_image, 2, 2, 10, 3)
    conv1 = conv.forward()
    print ('first convolution', conv1.shape)
    pool = Pooling(conv1, 2, 1)
    pool1 = pool.pool_forward()
    print ('first max pool', pool1.shape)
    conv = Convolution(pool1, 2, 2, 10, 3)
    conv2 = conv.forward()
    print ('second convolution', conv2.shape)
    pool = Pooling(conv2, 2, 1)
    pool2 = pool.pool_forward()
    print ('second max pool', pool2.shape)
    dense = Dense(pool2, 120, layer_type='hdden')
    dense1 = dense.forward()
    print ('first dense layer', dense1.shape)
    dense = Dense(dense1, 80, layer_type='last')
    dense2 = dense.forward()
    print ('last dense layer', dense2.shape)
    softmax = Softmax(dense2, 6) # 6 are the number of classes
    activation = softmax.forward()
    print ('softmax layer', activation)
