#!/usr/bin/env python3
""" 3. Strided Convolution """


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ performs a convolution on grayscale images """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        pad_h, pad_w = 0, 0
    elif padding == 'same':
        pad_h = (((h - 1) * sh + kh - h) // 2) + 1
        pad_w = (((w - 1) * sw + kw - w) // 2) + 1
    elif isinstance(padding, tuple):
        pad_h, pad_w = padding

    pad_img = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)))

    output_h = ((h + 2 * pad_h - kh) // sh) + 1
    output_w = ((w + 2 * pad_w - kw) // sw) + 1

    convoled = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            convoled[:, i, j] = np.sum(
                pad_img[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
                * kernel, axis=(1, 2))

    return convoled
