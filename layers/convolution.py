import numpy as np


class Convolution(object):

    def __init__(self, input_dim, pad, stride, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pad = pad
        self.stride = stride
        self.input_dim = input_dim
        self.W = np.random.normal(  # Xavier Initialization
            loc=0.0, scale=np.sqrt(
                2 / ((self.filter_size * self.filter_size * self.input_dim[-1]))),
            size=(self.filter_size, self.filter_size, self.input_dim[-1], self.num_filters))
        self.b = np.zeros(shape=(1, 1, 1, self.num_filters))
        self.n_H = int((self.input_dim[0] + 2*self.pad - self.filter_size)/self.stride) + 1
        self.n_W = int((self.input_dim[1] + 2*self.pad - self.filter_size)/self.stride) + 1
        self.n_C = num_filters
        self.output_dim = (self.n_H, self.n_W, self.n_C)
        self.params = [self.W, self.b]

    def zero_pad(self, X, pad):
        """
        Pads with zeros all images of the dataset X. Zeros are added around the
        border of an image.
        Parameters:
        X -- Image -- matrix of shape (m, n_H, n_W, n_C)
        pad -- padding amount -- int

        Returns:
        X_pad -- Image padded with zeros around width and height. -- matrix of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """

        X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        return X_pad

    def convolve(self, image_slice, W, b):
        """
        Apply a filter defined by W on a single slice of an image.

        Parameters:
        image_slice -- slice of input data of shape (f, f, n_C_prev)
        W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
        b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

        Returns:
        Z -- a scalar value, result of convolving the sliding window (W, b) on image_slice
        """

        Z = np.sum(np.multiply(image_slice, W)) + b  # ( W*X + b )
        return Z

    def get_corners(self, height, width, filter_size, stride):
        vert_start = height * stride
        vert_end = vert_start + filter_size
        horiz_start = width * stride
        horiz_end = horiz_start + filter_size
        return vert_start, vert_end, horiz_start, horiz_end

    def forward(self, A_prev):
        """
        Forward proporgation for convolution. This takes activations
        from previous layer and then convolve it with a filter defined by W with bias
        b.

        Parameters:

        Returns:

        """
        filter_size, filter_size, n_C_prev, n_C = self.W.shape
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        Z = np.empty((m, self.n_H, self.n_W, self.n_C))
        A_prev_pad = self.zero_pad(A_prev, self.pad)
        for i in range(m):
            a_prev_pad = A_prev_pad[i, :, :, :]
            for h in range(0, a_prev_pad.shape[0]):
                for w in range(0, a_prev_pad.shape[1]):
                    for c in range(n_C):
                        vert_start, vert_end, horiz_start, horiz_end = self.get_corners(
                            h, w, self.filter_size, self.stride)
                        if horiz_end <= a_prev_pad.shape[1] and vert_end <= a_prev_pad.shape[0]:
                            a_slice_prev = a_prev_pad[
                                vert_start:vert_end, horiz_start:horiz_end, :]
                            Z[i, h, w, c] = self.convolve(
                                a_slice_prev, self.W[:, :, :, c], self.b[:, :, :, c])
        assert (Z.shape == (m, self.n_H, self.n_W, self.n_C))
        return Z

    def backward(self, dZ):
        """
        Backward proporgation for convolution.

        Parameters:

        Returns:
        """
        n_H_prev, n_W_prev, n_C_prev = self.A_prev.shape
        f, f, n_C_prev, n_C = self.W.shape
        n_H, n_W, n_C = dZ.shape
        dA_prev = np.random.randn(n_H_prev, n_W_prev, n_C_prev)
        dW = np.random.randn(f, f, n_C_prev, n_C)
        db = np.zeros(shape=(1, 1, 1, self.num_filters))

        # Pad A_prev and dA_prev
        A_prev_pad = self.zero_pad(self.A_prev, self.pad)
        dA_prev_pad = self.zero_pad(dA_prev, self.pad)

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start, vert_end, horiz_start, horiz_end = self.get_corners(
                        h, w, self.filter_size, self.stride)
                    if horiz_end <= A_prev_pad.shape[1] and vert_end <= A_prev_pad.shape[0]:  # bounds
                        a_slice_prev = A_prev_pad[
                            vert_start:vert_end, horiz_start:horiz_end, :]
                        dA_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.W[:, :, c] * dZ[h, w, c]
                        dW[:, :, c] += a_slice_prev * dZ[h, w, c]
                        db[:, :, c] += dZ[h, w, c]

        dA_prev[:, :, :] = dA_prev_pad[self.pad:-self.pad, self.pad:-self.pad, :]
        assert(dA_prev.shape == (n_H_prev, n_W_prev, n_C_prev))

        return dA_prev, [dW, db]
