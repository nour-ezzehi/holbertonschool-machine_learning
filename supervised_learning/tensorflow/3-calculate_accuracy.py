#!/usr/bin/env python3
""" 3. Accuracy """


import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of a prediction
    """
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy
