import numpy as np
from layers.convolution import Convolution
from layers. pooling import Maxpool

def test_conv_step():

    np.random.seed(1)

    A_prev = np.random.randn(4, 4, 3)
    W = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)
    input_dim = A_prev.shape
    conv = Convolution(input_dim=input_dim, pad=2, stride=2,
                       num_filters=8,
                       filter_size=2, seed=1)
    Z = conv.convolve(A_prev, W, b)
    assert (round(Z, 11) == -6.99908945068)

def test_zero_pad():
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    input_dim = x.shape[1:]
    conv = Convolution(input_dim=input_dim, pad=2, stride=2,
                       num_filters=8,
                       filter_size=2, seed=1)
    x_pad = conv.zero_pad(x, 2)
    assert (x_pad.shape == (4, 7, 7, 2))
    assert (np.sum(x_pad[1, 1]) == 0)

def test_conv():

    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    input_dim = A_prev.shape[1:]
    conv = Convolution(input_dim=input_dim, pad=2, stride=2,
                       num_filters=8,
                       filter_size=2, seed=1)
    conv.params[0] = W
    conv.params[1] = b
    Z = conv.forward(A_prev)
    target = np.empty(8)
    target[:] = [-0.61490741, -6.7439236, -2.55153897, 1.75698377, 3.56208902,
                     0.53036437, 5.18531798, 8.75898442]
    hparameters = {"pad": 2,
                   "stride": 2}
    cache_conv = (A_prev, conv.params[0], conv.params[1], hparameters)
    target_cache_conv = np.empty(3)
    target_cache_conv = [-0.20075807, 0.18656139, 0.41005165]

    dA, grads = conv.backward(Z)
    dW = grads[0]
    db = grads[1]

    assert (round(np.mean(Z), 13) == 0.0489952035289)
    assert ((np.sum(np.around(Z[3, 2, 1], 8) != target)) == 0)
    assert ((np.sum(np.around(cache_conv[0][1][2][3], 8) != target_cache_conv)) == 0)

    assert(round(np.mean(dA), 11) == 1.45243777754)
    assert(round(np.mean(dW), 11) == 1.72699145831)
    assert(round(np.mean(db), 11) == 7.83923256462)

def test_pool_forward():

    np.random.seed(1)
    A_prev = np.random.randn(2, 4, 4, 3)
    maxpool1 = Maxpool(input_dim=(4, 4, 3),
                       filter_size=3,
                       stride=2)
    Z = maxpool1.forward(A_prev)
    target_Z = np.empty((2, 1, 1, 3))
    target_Z[0] = [1.74481176, 0.86540763, 1.13376944]
    target_Z[1] = [1.13162939, 1.51981682,  2.18557541]


    assert ((np.array_equal(np.around(Z, 8), target_Z)) == True)

def test_pool_mask():

    np.random.seed(1)
    x = np.random.randn(2, 3)
    maxpool1 = Maxpool(input_dim=(4, 4, 3),
                       filter_size=3,
                       stride=2)
    target_x = np.empty((2, 3))
    target_x[0] = [1.62434536, -0.61175641, -0.52817175]
    target_x[1] = [-1.07296862, 0.86540763, -2.3015387]


    mask = maxpool1.create_mask_from_window(x)
    target_mask = np.empty((2, 3))
    target_mask[0] = [True, False, False]
    target_mask[1] = [False, False, False]


    assert np.array_equal(target_x, np.around(x, 8)) == True
    assert np.array_equal(target_mask, mask) == True

def test_pool():

    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    maxpool1 = Maxpool(input_dim=(5, 3, 2),
                       filter_size=2,
                       stride=1)
    Z = maxpool1.forward(A_prev)
    dA = np.random.randn(5, 4, 2, 2)
    dA_prev, _ = maxpool1.backward(dA)

    target_dA_prev = np.empty((3, 2))
    target_dA_prev[0] = [0, 0]
    target_dA_prev[1] = [5.05844394, -1.68282702]
    target_dA_prev[2] = [0, 0]


    assert (round(np.mean(dA), 12) == 0.145713902729)
    assert np.array_equal(np.around(dA_prev[1, 1], 8), target_dA_prev) == True






