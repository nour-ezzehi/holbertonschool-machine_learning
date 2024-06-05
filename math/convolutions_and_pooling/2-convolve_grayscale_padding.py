#!/usr/bin/env python3
""" 2. Convolution with Padding """


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ performs a convolution on grayscale images with custom padding """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1
    padding_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    convoled = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):

            image_slice = padding_images[:, i:i + kh, j:j + kw]
            convoled[:, i, j] = np.sum(image_slice * kernel, axis=(1, 2))

    return convoled
