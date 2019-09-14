import numpy as np


class Convolution(object):

    def __init__(self, input_dim, pad, stride, num_filters, filter_size, seed):
        self.seed = seed
        np.random.seed(self.seed)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pad = pad
        self.stride = stride
        self.input_dim = input_dim
        self.n_H = int((self.input_dim[0] + 2 * self.pad - self.filter_size) / self.stride) + 1
        self.n_W = int((self.input_dim[1] + 2 * self.pad - self.filter_size) / self.stride) + 1
        self.n_C = num_filters
        self.output_dim = (self.n_H, self.n_W, self.n_C)
        self.W = np.random.randn(
            self.filter_size, self.filter_size, self.input_dim[-1], num_filters) * np.sqrt(2 / np.prod(self.input_dim)
                                                                                           + np.prod(self.output_dim))
        #self.W = np.random.normal(  # Xavier Initialization
        #    loc=0.0, scale=np.sqrt(
        #        2 / ((self.filter_size * self.filter_size * self.input_dim[-1]))),
        #    size=(self.filter_size, self.filter_size, self.input_dim[-1], self.num_filters))
        self.b = np.zeros(shape=(1, 1, 1, self.num_filters))

        self.params = [self.W, self.b]

    def zero_pad(self, X, pad):
        """
        Set padding to the image X.

        Pads with zeros all images of the dataset X.
        Zeros are added around the border of an image.

        Parameters:
        X -- Image -- numpy array of shape (m, n_H, n_W, n_C)
        pad -- padding amount -- int

        Returns:
        X_pad -- Image padded with zeros around width and height. -- numpy array of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)

        """
        X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        return X_pad

    def convolve(self, image_slice, W, b):
        """
        Apply a filter defined by W on a single slice of an image.

        Parameters:
        image_slice -- slice of input data -- numpy array of shape (f, f, n_C_prev)
        W -- Weight parameters contained in a window - numpy array of shape (f, f, n_C_prev)
        b -- Bias parameters contained in a window - numpy array of shape (1, 1, 1)

        Returns:
        Z -- a scalar value, result of convolving the sliding window (W, b) on image_slice

        """
        s = np.multiply(image_slice, W)
        z = np.sum(s)
        Z = z + float(b)
        return Z

    def get_corners(self, height, width, filter_size, stride):
        """
        Get corners of the image relative to stride.

        Parameters:
        height -- height of an image -- int
        width -- width of an image -- int
        filter_size -- size of filter -- int
        stride -- amount by which the filter shifts -- int

        Returns:
        vert_start -- a scalar value, upper left corner of the box.
        vert_end -- a scalar value, upper right corner of the box.
        horiz_start -- a scalar value, lower left corner of the box.
        horiz_end -- a scalar value, lower right corner of the box.

        """
        vert_start = height * stride
        vert_end = vert_start + filter_size
        horiz_start = width * stride
        horiz_end = horiz_start + filter_size
        return vert_start, vert_end, horiz_start, horiz_end


    def forward(self, A_prev):
        """
        Forward proporgation for convolution.

        This takes activations from previous layer and then convolve it
        with a filter defined by W with bias b.

        Parameters:
        A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        Z -- convolution output, numpy array of shape (m, n_H, n_W, n_C)

        """
        np.random.seed(self.seed)
        self.A_prev = A_prev
        filter_size, filter_size, n_C_prev, n_C = self.params[0].shape
        m, n_H_prev, n_W_prev, n_C_prev = self.A_prev.shape
        Z = np.zeros((m, self.n_H, self.n_W, self.n_C))
        A_prev_pad = self.zero_pad(self.A_prev, self.pad)
        for i in range(m):
            a_prev_pad = A_prev_pad[i, :, :, :]
            for h in range(self.n_H):
                for w in range(self.n_W):
                    for c in range(n_C):
                        vert_start, vert_end, horiz_start, horiz_end = self.get_corners(
                            h, w, self.filter_size, self.stride)
                        #if horiz_end <= a_prev_pad.shape[1] and vert_end <= a_prev_pad.shape[0]:
                        a_slice_prev = a_prev_pad[
                                vert_start:vert_end, horiz_start:horiz_end, :]
                        Z[i, h, w, c] = self.convolve(
                                a_slice_prev, self.params[0][:, :, :, c], self.params[1][:, :, :, c])

        assert (Z.shape == (m, self.n_H, self.n_W, self.n_C))
        return Z

    def backward(self, dZ):
        """
        Backward proporgation for convolution.

        Parameters:
        dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)

        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv
                   layer (A_prev), numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
                  numpy array of shape (f, f, n_C_prev, n_C)
        db -- gradient of the cost with respect to the biases of the conv layer (b)
                  numpy array of shape (1, 1, 1, n_C)

        """
        np.random.seed(self.seed)
        m, n_H_prev, n_W_prev, n_C_prev = self.A_prev.shape
        f, f, n_C_prev, n_C = self.params[0].shape
        m, n_H, n_W, n_C = dZ.shape
        dA_prev = np.zeros(self.A_prev.shape)
        dW = np.zeros(self.params[0].shape)
        db = np.zeros(self.params[1].shape)
        # Pad A_prev and dA_prev
        A_prev_pad = self.zero_pad(self.A_prev, self.pad)
        dA_prev_pad = self.zero_pad(dA_prev, self.pad)
        for i in range(m):
            a_prev_pad = A_prev_pad[i, :, :, :]
            da_prev_pad = dA_prev_pad[i, :, :, :]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start, vert_end, horiz_start, horiz_end = self.get_corners(
                            h, w, self.filter_size, self.stride)
                        #if horiz_end <= a_prev_pad.shape[1] and vert_end <= a_prev_pad.shape[0]:  # bounds
                        a_slice_prev = a_prev_pad[
                                vert_start:vert_end, horiz_start:horiz_end, :]
                        da_prev_pad[
                                vert_start:vert_end, horiz_start:horiz_end, :] += self.params[0][:, :, :, c] * dZ[i, h, w, c]
                        dW[:, :, :, c] += a_slice_prev * dZ[i, h, w, c]
                        db[:, :, :, c] += dZ[i, h, w, c]

            dA_prev[i, :, :, :] = da_prev_pad[self.pad:-self.pad, self.pad:-self.pad, :]
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

        return dA_prev, [dW, db]

