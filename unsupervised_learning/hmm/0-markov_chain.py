#!/usr/bin/env python3
""" 0. Markov Chain """
import numpy as np


def markov_chain(P, s, t=1):
    """
    determines the probability of a markov chain being in a particular state
    after a specified number of iterations
    """

    if not isinstance(P, np.ndarray) or not isinstance(t, int):
        return None
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or t <= 0:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None

    for _ in range(t):
        s = np.matmul(s, P)

    return s
