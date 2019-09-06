import numpy as np


class Convolution(object):

    def __init__(self, A_prev, pad, stride, num_filters, filter_size):
        np.random.seed(1)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pad = pad
        self.stride = stride
        self.A_prev = A_prev
        self.W = np.random.normal(  # Xavier Initialization
            loc=0.0, scale=np.sqrt(
                2 / ((filter_size * filter_size * self.A_prev.shape[-1]))),
            size=(filter_size, filter_size, self.A_prev.shape[-1], num_filters))
        self.b = np.zeros(shape=(1, 1, 1, num_filters))

    def zero_pad(self, X, pad):
        """
        Pads with zeros all images of the dataset X. Zeros are added around the
        border of an image.
        Parameters:
        X -- Image -- matrix of shape (n_W, n_H, n_C)
        pad -- padding amount -- int

        Returns:
        X_pad -- Image padded with zeros around width and height. -- matrix of shape (m, n_W + 2*pad, n_H + 2*pad, n_C)
        """

        X_pad = np.pad(X, ((pad, pad), (pad, pad), (0, 0)), 'constant')
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

    def slice_region(self, A_prev, height, width, filter_size, stride):
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
                vert_start:vert_end, horiz_start:horiz_end, :]
            return a_slice_prev

    def relu(self, Z):
        A = np.maximum(0, Z)  # element-wise
        return A

    def forward(self):
        """
        Forward proporgation for convolution layer. This layer takes activations
        from previous layer and then convolve it with a filter defined by W with bias
        b.

        Parameters:

        Returns:

        """

        filter_size, filter_size, n_C_prev, n_C = self.W.shape
        n_H_prev, n_W_prev, n_C_prev = self.A_prev.shape
        n_H = int((n_H_prev + 2*self.pad - filter_size)/self.stride) + 1
        n_W = int((n_W_prev + 2*self.pad - filter_size)/self.stride) + 1
        Z = np.empty((n_H, n_W, n_C))
        a_prev_pad = self.zero_pad(self.A_prev, self.pad)
        for h in range(0, a_prev_pad.shape[0]):
            for w in range(0, a_prev_pad.shape[1]):
                for c in range(n_C):
                    region = self.slice_region(
                        a_prev_pad, h, w, filter_size, self.stride)
                    if region is not None:
                        Z[h, w, c] = self.convolve(
                            region, self.W[:, :, :, c], self.b[:, :, :, c])

        assert (Z.shape == (n_H, n_W, n_C))
        A = self.relu(Z)
        assert(np.isnan(A).sum() == 0)
        assert (A.shape == (n_H, n_W, n_C))
        return A

    def relu_backward(self):
        return

    def backward(self, A_prev, W, b, hparameters):
        """
        Forward proporgation for convolution layer. This layer takes activations
        from previous layer and then convolve it with a filter defined by W with bias
        b.

        Parameters:

        Returns:
        """

        return
