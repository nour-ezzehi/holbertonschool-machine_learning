#!/usr/bin/env python3
""" 4. Loss """


import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """ calculates the softmax cross-entropy loss of a prediction
    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                          (labels=y, logits=y_pred))

    return loss
