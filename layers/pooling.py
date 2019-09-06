import numpy as np


class Pooling(object):

    def __init__(self, A_prev, filter_size, stride, mode='max'):
        self.filter_size = filter_size
        self.stride = stride
        self.A_prev = A_prev
        self.mode = mode

    def slice_region(self, A_prev, height, width, channel, filter_size, stride):
        """
        Cuts a region of activations based on given stride and filter size.

        Parameters
        height -- height of an image -- int
        widht -- width of an image -- int
        filter_size -- size of convolution filter -- int
        stride = stride for convolution filter -- int

        Returns:
        a_slice_prev -- slice of input activations -- matrix of shape ()
        """

        vert_start = height * stride
        vert_end = vert_start + filter_size
        horiz_start = width * stride
        horiz_end = horiz_start + filter_size
        if horiz_end <= A_prev.shape[1] and vert_end <= A_prev.shape[0]:
            a_slice_prev = A_prev[
                vert_start:vert_end, horiz_start:horiz_end, channel]
            return a_slice_prev

    def pool_forward(self):
        n_H_prev, n_W_prev, n_C_prev = self.A_prev.shape
        n_H = int((n_H_prev - self.filter_size)/self.stride) + 1
        n_W = int((n_W_prev - self.filter_size)/self.stride) + 1
        Z = np.random.randn(n_H, n_W, n_C_prev)
        for h in range(0, self.A_prev.shape[0]):
            for w in range(0, self.A_prev.shape[1]):
                for c in range(n_C_prev):
                    region = self.slice_region(
                        self.A_prev, h, w, c, self.filter_size, self.stride)
                    if region is not None:
                        if self.mode == 'max':
                            Z[h, w, c] = np.max(region)
                        else:
                            Z[h, w, c] = np.mean(region)
        assert(Z.shape == (n_H, n_W, n_C_prev))
        return Z

    def pool_backward(self):
        return 0
