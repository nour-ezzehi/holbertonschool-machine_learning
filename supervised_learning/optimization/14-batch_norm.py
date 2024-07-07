#!/usr/bin/env python3
""" 14.Batch Normalization Upgraded """


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ creates a batch normalization layer for a
    neural network in tensorflow """

    dense_layer = tf.keras.layers.Dense(
        units=n, kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'))

    z = dense_layer(prev)
    gamma = tf.Variable(initial_value=tf.ones((1, n)), name='gamma')
    beta = tf.Variable(initial_value=tf.zeros((1, n)), name='beta')
    mean, variance = tf.nn.moments(Z, axes=[0])
    epsilon = 1e-7

    batch_norm_layer = tf.nn.BatchNormalization(
        z, mean, variance, beta, gamma, epsilon)

    activated_output = activation(batch_norm_layer)

    return activated_output
