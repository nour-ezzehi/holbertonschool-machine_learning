#!/usr/bin/env python3
""" 3. Optimize k """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ tests for the optimum number of clusters by variance """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    if type(kmin) is not int or kmin < 1:
        return None, None
    if kmax is not None and (type(kmax) is not int or kmax < 1):
        return None, None
    if kmax is not None and kmin >= kmax:
        return None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    results = list()
    d_vars = list()

    for i in range(kmin, kmax + 1):
        C, clss = kmeans(X, i, iterations=1000)
        results.append((C, clss))

        if i == kmin:
            first_var = variance(X, C)
        var = variance(X, C)
        d_vars.append(first_var - var)

    return results, d_vars
