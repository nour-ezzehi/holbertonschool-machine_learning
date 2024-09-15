#!/usr/bin/env python3
""" 2. Absorbing Chains """
import numpy as np


def absorbing(P):
    """ determines if a markov chain is absorbing """

    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    num_states = P.shape[0]
    if not np.isclose(np.sum(P, axis=1), np.ones(num_states))[0]:
        return None

    if np.all(np.diag(P) != 1):
        return False

    if np.all(np.diag(P) == 1):
        return True

    for i in range(num_states):
        if np.any(P[i, :] == 1):
            continue
        break

    sub_mat_I = P[:i, :i]
    id_mat = np.identity(num_states - i)
    sub_mat_R = P[i:, :i]
    sub_mat_Q = P[i:, i:]

    try:
        fundamental_matrix = np.linalg.inv(id_mat - sub_mat_Q)
    except Exception:
        return False

    FR_product = np.matmul(fundamental_matrix, sub_mat_R)
    limiting_matrix = np.zeros((num_states, num_states))
    limiting_matrix[:i, :i] = sub_mat_I
    limiting_matrix[i:, :i] = FR_product

    sub_mat_Qbar = limiting_matrix[i:, i:]

    if np.all(sub_mat_Qbar == 0):
        return True

    return False
