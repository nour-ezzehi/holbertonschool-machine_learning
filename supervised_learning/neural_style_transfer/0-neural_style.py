#!/usr/bin/env python3
""" 0. Initialize """


import numpy as np
import tensorflow as tf


class NST:
    """The NST class performs tasks for neural style transfer
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ NST constructor """
        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (float, int)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (float, int)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels.
        """

        if (not isinstance(image, np.ndarray) or image.shape[-1] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w = image.shape[:2]

        if w > h:
            new_w = 512
            new_h = int((h * 512) / w)
        else:
            new_h = 512
            new_w = int((w * 512) / h)

        image_resized = tf.image.resize(
            image, size=[new_h, new_w],
            method=tf.image.ResizeMethod.BICUBIC)

        image_normalized = image_resized / 255
        image_clipped = tf.clip_by_value(image_normalized, 0, 1)

        return tf.expand_dims(image_clipped, axis=0)
