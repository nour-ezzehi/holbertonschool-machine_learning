#!/usr/bin/env python3
""" 14.Batch Normalization Upgraded """


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ creates a batch normalization layer for a
    neural network in tensorflow """

    dense_layer = tf.keras.layers.Dense(
        units=n, kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'))(prev)

    batch_norm_layer = tf.keras.layers.BatchNormalization(
        epsilon=1e-7, gamma_initializer=tf.keras.initializers.Ones(),
        beta_initializer=tf.keras.initializers.Zeros()
    )(dense_layer)

    activated_output = activation(batch_norm_layer)

    return activated_output
