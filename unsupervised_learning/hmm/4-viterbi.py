#!/usr/bin/env python3
""" 4. The Viretbi Algorithm """
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ calculates the most likely sequence of hidden states
    for a hidden markov model"""

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
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        for j in range(N):
            F[j, i] = np.max(F[:, i - 1] * Transition[:, j] *
                             Emission[j, Observation[i]])
    P = np.max(F[:, -1])
    path = np.zeros(T, dtype=int)
    path[T - 1] = np.argmax(F[:, -1])

    for i in range(T - 2, -1, -1):
        path[i] = np.argmax(F[:, i] * Transition[:, path[i + 1]])

    return path, P
