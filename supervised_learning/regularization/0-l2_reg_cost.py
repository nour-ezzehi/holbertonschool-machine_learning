#!/usr/bin/env python3
"""L2 Regularization Cost"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ calculates the cost of a neural network with L2 regularization
    """
    l2_reg_term = 0
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        l2_reg_term += np.sum(np.square(W))

    l2_reg_cost = cost + (lambtha / (2 * m)) * l2_reg_term

    return l2_reg_cost
