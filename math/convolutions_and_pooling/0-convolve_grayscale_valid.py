#!/usr/bin/env python3
""" 0. Valid Convolution """


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ performs a valid convolution on grayscale images """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1

    convoluted_images = np.zeros((m, output_h, output_w))

    for h in range(h - kh + 1):
        for w in range(w - kw + 1):
            output = np.sum(images[:, h: h + kh, w: w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted_images[:, h, w] = output

    return convoluted_images
