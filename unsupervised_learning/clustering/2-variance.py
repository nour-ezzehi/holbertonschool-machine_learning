#!/usr/bin/env python3
""" 2. Variance """
import numpy as np


def variance(X, C):
    """
    calculates the total intra-cluster variance for a data set
    """

    try:
        n, d = X.shape
        k, d = C.shape
        distance = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clusters = np.argmin(distance, axis=1)

        return np.sum((X - C[clusters]) ** 2)

    except Exception:
        return None
