#!/usr/bin/env python3
""" 12. Learning Rate Decay Upgraded """


import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates a learning rate decay operation in tensorflow
    using inverse time decay
    """

    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)
