#!/usr/bin/env python3
""" 5. The Backward Algorithm """
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ performs the backward algorithm for a hidden markov model
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None
    B = np.zeros((N, T))
    B[:, T - 1] = 1

    for i in range(T - 2, -1, -1):
        for j in range(N):
            B[j, i] = np.sum(B[:, i + 1] * Transition[j, :] *
                             Emission[:, Observation[i + 1]])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

    return P, B
