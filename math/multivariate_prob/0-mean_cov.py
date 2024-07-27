#!/usr/bin/env python3
""" 0. Mean and Covariance """
import numpy as np


def mean_cov(X):
    """ calculates the mean and covariance of a data set"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    n = X.shape[0]
    if n < 2:
        raise ValueError('X must contain multiple data points')

    mean = np.mean(X, axis=0)
    cov = np.dot((X - mean).T, X - mean) / (n - 1)

    return mean.reshape(1, -1), cov
