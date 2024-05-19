#!/usr/bin/env python3
"""6. Momentum Upgraded """


import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """sets up the gradient descent with momentum
    optimization algorithm in TensorFlow
    """

    optimizer = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)

    return optimizer
