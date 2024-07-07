#!/usr/bin/env python3
""" 13. Batch Normalization """


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ normalizes an unactivated output of a
    neural network using batch normalization """

    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_normalized = (Z - mean) / np.sqrt(variance + epsilon)
    Z_scaled = gamma * Z_normalized + beta

    return Z_scaled
