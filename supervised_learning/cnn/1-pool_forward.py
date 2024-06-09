#!/usr/bin/env python3
""" 1. Pooling Forward Prop """


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs forward propagation over a pooling layer of a NN """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    A = np.zeros((m, h_new, w_new, c_prev))

    for h in range(h_new):
        for w in range(w_new):
            if mode == 'max':
                A[:, h, w, :] = np.max(
                    A_prev[:, h * sh:h * sh + kh, w * sw:w * sw + kw, :],
                    axis=(1, 2))
            elif mode == 'avg':
                A[:, h, w, :] = np.mean(
                    A_prev[:, h * sh:h * sh + kh, w * sw:w * sw + kw, :],
                    axis=(1, 2))

    return A
