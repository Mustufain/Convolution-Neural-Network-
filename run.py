from layers.convolution import Convolution
from layers.pooling import Pooling
from layers.dense import Dense
from layers.softmax_layer import Softmax
from utils.cnn_utils import load_dataset, convert_to_one_hot


"""
conv -> relu - > pool -> conv -> relu -> pool -> fc -> relu -> fc -> softmax
"""


if __name__ == '__main__':
    train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
    num_of_classes = len(classes)
    train_set_x = train_set_x / 255
    test_image = train_set_x[0]
    train_y = convert_to_one_hot(train_set_y, num_of_classes)
    test_label = train_y[:, 0]

    conv = Convolution(test_image, 2, 2, 10, 3)
    conv1 = conv.forward()
    pool = Pooling(conv1, 2, 1)
    pool1 = pool.pool_forward()
    conv = Convolution(pool1, 2, 2, 10, 3)
    conv2 = conv.forward()
    pool = Pooling(conv2, 2, 1)
    pool2 = pool.pool_forward()
    dense = Dense(pool2, 120, layer_type='hdden')
    dense1 = dense.forward()
    dense = Dense(dense1, 80, layer_type='last')
    dense2 = dense.forward()
    softmax = Softmax(dense2, 6) # 6 are the number of classes
    activation = softmax.forward()
    print ('cost', softmax.compute_cost(activation, test_label))
    print (softmax.compute_accuracy(activation, test_label))

    ## BACKPROPOGATION

    
