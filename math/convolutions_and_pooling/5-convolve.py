#!/usr/bin/env python3
""" 5. Multiple Kernels """


import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ performs a convolution on images using multiple kernels """
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride

    if padding == 'valid':
        pad_h, pad_w = 0, 0
    elif padding == 'same':
        pad_h = (((h - 1) * sh + kh - h) // 2) + 1
        pad_w = (((w - 1) * sw + kw - w) // 2) + 1
    elif isinstance(padding, tuple):
        pad_h, pad_w = padding
    padded = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))

    output_h = ((h + 2 * pad_h - kh) // sh) + 1
    output_w = ((w + 2 * pad_w - kw) // sw) + 1

    convoled = np.zeros((m, output_h, output_w, nc))
    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                convoled[:, i, j, k] = np.sum(
                    padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :] *
                    kernels[:, :, :, k], axis=(1, 2, 3))

    return convoled
