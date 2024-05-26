#!/usr/bin/env python3
"""2. L2 Regularization Cost"""


import tensorflow as tf


def l2_reg_cost(cost, model):
    """calculates the cost of a neural network with L2 regularization"""
    l2_reg_loss = list()
    for layer in model.layers:
        l2_reg_loss.append(tf.reduce_sum(layer.losses) + cost)

    return tf.stack(l2_reg_loss[1:])
