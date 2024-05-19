#!/usr/bin/env python3
"""8. RMSProp Upgraded
"""


import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """ sets up the RMSProp optimization algorithm in TensorFlow """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha, rho=beta2, epsilon=epsilon)
