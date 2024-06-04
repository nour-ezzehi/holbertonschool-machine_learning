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

    for i in range(output_h):
        for j in range(output_w):
            output = np.sum(images[:, i: i + kh, j: j + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted_images[:, i, j] = output

    return convoluted_images
