#!/usr/bin/env python3
""" 2. Shuffle Data """

import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way
    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    X_shuffled = X[shuffle]
    Y_shuffled = Y[shuffle]

    return X_shuffled, Y_shuffled
