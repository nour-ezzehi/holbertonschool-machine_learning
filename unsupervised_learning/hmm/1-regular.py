#!/usr/bin/env python3
""" 1. Regular Chains
"""


import numpy as np


def regular(P):
    """determines the steady state probabilities of a regular markov chain
    """

    if type(P) is not np.ndarray or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    regular = False

    for i in range(20):
        pw = np.linalg.matrix_power(P, i) > 0
        if False not in pw:
            regular = True
            break

    if regular:
        dim = P.shape[0]
        q = (P-np.eye(dim))
        ones = np.ones(dim)
        q = np.c_[q, ones]
        QTQ = np.dot(q, q.T)
        bQT = np.ones(dim)
        solved = np.linalg.solve(QTQ, bQT)
        if solved.ndim == 1:
            return solved[np.newaxis, ...]
        return solved
    else:
        return None
