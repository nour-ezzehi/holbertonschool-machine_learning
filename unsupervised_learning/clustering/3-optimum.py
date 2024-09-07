#!/usr/bin/env python3
""" 3. Optimize k """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ tests for the optimum number of clusters by variance """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < 1):
        return None, None
    if kmax is not None and kmin >= kmax:
        return None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    results = list()
    d_vars = list()

    C, clss = kmeans(X, kmin, iterations)
    if C is None or clss is None:
        return None, None

    results.append((C, clss))
    small_var = variance(X, C)
    d_vars.append(0.0)

    for k in range(kmin + 1, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        current_var = variance(X, C)
        d_vars.append(small_var - current_var)
        results.append((C, clss))

    return results, d_vars
