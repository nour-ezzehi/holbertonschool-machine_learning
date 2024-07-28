#!/usr/bin/env python3
""" 1. Correlation """
import numpy as np


def correlation(C):
    """ calculates a correlation matrix """
    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a numpy.ndarray')
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    std_dev = np.sqrt(np.diag(C))
    d = C.shape[0]
    corr = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            corr[i, j] = C[i, j] / (std_dev[i] * std_dev[j])

    return corr
