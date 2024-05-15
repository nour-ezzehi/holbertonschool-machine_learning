#!/usr/bin/env python3
""" 0. Normalization Constants """

import numpy as np


def normalization_constants(X):
    """calculates the normalization (standardization) constants of a matrix
    """
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)

    return mean, stdev
