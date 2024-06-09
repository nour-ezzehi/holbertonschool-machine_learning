#!/usr/bin/env python3
""" 3. Pooling Back Prop """


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ that performs back propagation over
    a pooling layer of a neural network
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    m, h_new, w_new, c_new = dA.shape
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw
                    if mode == 'max':
                        a_slice = a_prev[v_start:v_end, h_start:h_end, c]
                        mask = (a_slice == np.max(a_slice))
                        dA_prev[i, v_start:v_end, h_start:h_end, c] +=\
                            np.multiply(mask, dA[i, h, w, c])
                    else:
                        da = dA[i, h, w, c]
                        shape = kernel_shape
                        avg = da / (kh * kw)
                        Z = np.ones(shape) * avg
                        dA_prev[i, v_start:v_end, h_start:h_end, c] += Z

    return dA_prev
