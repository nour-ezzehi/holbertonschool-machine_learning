#!/usr/bin/env python3
""" 3. PDF """
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

    def pdf(self, x):
        """ calculates the PDF at a data point """
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError(f'x must have the shape ({d}, 1)')

        cov_inv = np.linalg.inv(self.cov)
        cov_det = np.linalg.det(self.cov)
        exp_term = -0.5 * \
            np.dot(np.dot((x - self.mean).T, cov_inv), (x - self.mean))
        norm_term = 1 / (((2 * np.pi) ** (d / 2)) * (cov_det ** 0.5))

        return float(np.exp(exp_term) * norm_term)
