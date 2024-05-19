#!/usr/bin/env python3
"""6. Momentum Upgraded """


import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    creating the whole
    momentum optimization process
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)

    return optimizer
