import numpy as np


class Maxpool(object):

    def __init__(self, input_dim, filter_size, stride):
        self.filter_size = filter_size
        self.stride = stride
        self.input_dim = input_dim
        self.n_H = int((input_dim[0] - self.filter_size)/self.stride) + 1
        self.n_W = int((input_dim[1] - self.filter_size)/self.stride) + 1
        self.n_C = input_dim[-1]
        self.output_dim = (self.n_H, self.n_W, self.n_C)
        self.params = []

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
        Forward prpogation of the pooling layer.

        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        Z -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)

        """
        self.A_prev = A_prev
        m, n_H_prev, n_W_prev, n_C_prev = self.A_prev.shape
        Z = np.random.randn(m, self.n_H, self.n_W, n_C_prev)
        for i in range(0, m):
            a_prev = self.A_prev[i]
            for h in range(0, a_prev.shape[0]):
                for w in range(0, a_prev.shape[1]):
                    for c in range(n_C_prev):
                        vert_start, vert_end, horiz_start, horiz_end = self.get_corners(
                            h, w, self.filter_size, self.stride)
                        if horiz_end <= a_prev.shape[1] and vert_end <= a_prev.shape[0]:
                            a_slice_prev = a_prev[
                                vert_start:vert_end, horiz_start:horiz_end, c]
                            Z[i, h, w, c] = np.max(a_slice_prev)
        assert(Z.shape == (m, self.n_H, self.n_W, n_C_prev))
        return Z

    def create_mask_from_window(self, image_slice):
        """
        Get  mask from a image_slice to identify the max entry.

        Parameters:
        image_slice -- numpy array of shape (f, f, n_C_prev)

        Returns:
        mask -- Array of the same shape as window, contains a True at the
                position corresponding to the max entry of iamge_slice.

        """
        mask = np.max(image_slice)
        mask = (image_slice == mask)
        return mask

    def backward(self, dA):
        """
        Backward propogation of the pooling layer.

        Parameters:
        dA -- gradient of cost with respect to the output of the pooling layer,
              same shape as Z

        Returns:
        dA_prev -- gradient of cost with respect to the input of the pooling layer,
                   same shape as A_prev

        """
        m, n_H_prev, n_W_prev, n_C_prev = self.A_prev.shape
        m, n_H, n_W, n_C = dA.shape
        dA_prev = np.random.randn(m, n_H_prev, n_W_prev, n_C_prev)
        for i in range(m):
            a_prev = self.A_prev[i]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start, vert_end, horiz_start, horiz_end = self.get_corners(
                            h, w, self.filter_size, self.stride)
                        if horiz_end <= a_prev.shape[1] and vert_end <= a_prev.shape[0]:  # bounds
                            a_prev_slice = a_prev[
                                vert_start:vert_end, horiz_start:horiz_end, c]
                            mask = self.create_mask_from_window(a_prev_slice)
                            dA_prev[
                                i, vert_start: vert_end, horiz_start: horiz_end,
                                c] += np.multiply(mask, dA[i, h, w, c])
        assert(dA_prev.shape == self.A_prev.shape)
        return dA_prev, []
