#!/usr/bin/env python3
""" 2. Initialize """
import numpy as np


def mean_cov(X):
    """ calculates the mean and covariance of a data set"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    n = X.shape[1]
    if n < 2:
        raise ValueError('X must contain multiple data points')

    mean = np.mean(X, axis=1, keepdims=True)
    cov = np.dot((X - mean), (X - mean).T) / (n - 1)

    return mean, cov


class MultiNormal:
    """ class MultiNormal: epresents a Multivariate Normal distribution """

    def __init__(self, data):
        """ constructor function """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        n = data.shape[1]
        if n < 2:
            raise ValueError('data must contain multiple data points')
        self.mean, self.cov = mean_cov(data)
