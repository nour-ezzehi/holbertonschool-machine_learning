#!/usr/bin/env python3
""" 6. Pooling """


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ performs pooling on images """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = ((h - kh) // sh) + 1
    output_w = ((w - kw) // sw) + 1
    convoled = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            if mode == 'max':
                convoled[:, i, j, :] = np.max(
                    images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2))
            elif mode == 'avg':
                convoled[:, i, j, :] = np.mean(
                    images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    axis=(1, 2))

    return convoled
