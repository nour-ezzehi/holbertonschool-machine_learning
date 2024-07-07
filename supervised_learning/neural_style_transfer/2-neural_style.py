#!/usr/bin/env python3
""" 2. Gram Matrix """


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
        self.load_model()

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

    def load_model(self):
        """ Load the VGG19 model """

        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        vgg.trainable = False
        for layer in vgg.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer.__class__ = tf.keras.layers.AveragePooling2D

        style_outputs = [vgg.get_layer(name).output
                         for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        self.model = tf.keras.models.Model(
            inputs=vgg.input,
            outputs=style_outputs + [content_output])

    @staticmethod
    def gram_matrix(input_layer):
        """ Calculate the Gram matrix of a given tensor """

        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4
                or input_layer.shape[0] != 1):
            raise TypeError("input_layer must be a tensor of rank 4")
        gram_matrix = tf.linalg.einsum(
            'bijc,bijd->bcd', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        nb_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

        return gram_matrix / nb_locations
