#!/usr/bin/env python3
""" 6. The Baum-Welch Algorithm """
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ performs the forward algorithm for a hidden markov model
    """

    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None

    if np.all(np.isclose(np.sum(Emission, axis=1), 1)) is False:
        return None, None

    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    if Initial.shape[1] != 1:
        return None, None

    if np.all(np.isclose(np.sum(Initial), 1)) is False:
        return None, None

    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None

    if Transition.shape[0] != Initial.shape[0]:
        return None, None
    if Transition.shape[1] != Initial.shape[0]:
        return None, None

    if np.all(np.isclose(np.sum(Transition, axis=1), 1)) is False:
        return None, None

    hidden_states = Initial.shape[0]

    observations = Observation.shape[0]

    F = np.zeros((hidden_states, observations))

    index = Observation[0]

    E = Emission[:, index]

    F[:, 0] = Initial.T * E

    for i in range(1, observations):
        for j in range(hidden_states):
            F[j, i] = np.sum(F[:, i-1] * Transition[
                :, j] * Emission[j, Observation[i]])


    P = np.sum(F[:, observations-1], axis=0)

    return P, F


def backward(Observation, Emission, Transition, Initial):
    """ performs the backward algorithm for a hidden markov model
    """

    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None

    if np.all(np.isclose(np.sum(Emission, axis=1), 1)) is False:
        return None, None

    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    if Initial.shape[1] != 1:
        return None, None

    if np.all(np.isclose(np.sum(Initial), 1)) is False:
        return None, None

    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None

    if Transition.shape[0] != Initial.shape[0]:
        return None, None
    if Transition.shape[1] != Initial.shape[0]:
        return None, None

    if np.all(np.isclose(np.sum(Transition, axis=1), 1)) is False:
        return None, None

    hidden_states = Initial.shape[0]
    observations = Observation.shape[0]

    B = np.zeros((hidden_states, observations))

    B[:, observations - 1] = 1

    for t in range(observations - 2, -1, -1):
        for s in range(hidden_states):
            B[s, t] = np.sum(B[:, t + 1] * Transition[s, :] *
                             Emission[:, Observation[t + 1]])

    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

    return P, B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """ performs the Baum-Welch algorithm for a hidden markov mode
    """

    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None

    if np.all(np.isclose(np.sum(Emission, axis=1), 1)) is False:
        return None, None

    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None

    if Initial.shape[1] != 1:
        return None, None

    if np.all(np.isclose(np.sum(Initial), 1)) is False:
        return None, None

    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None

    if Transition.shape[0] != Initial.shape[0]:
        return None, None
    if Transition.shape[1] != Initial.shape[0]:
        return None, None

    if np.all(np.isclose(np.sum(Transition, axis=1), 1)) is False:
        return None, None

    if not isinstance(iterations, int) or iterations < 0:
        return None, None

    hidden_states = Initial.shape[0]

    observations = Observations.shape[0]

    output_states = Emission.shape[1]

    transition_prev = Transition.copy()
    emission_prev = Emission.copy()

    for iteration in range(iterations):
        _, F = forward(Observations, Emission, Transition, Initial)

        _, B = backward(Observations, Emission, Transition, Initial)

        NUM = np.zeros((hidden_states, hidden_states, observations - 1))


        for t in range(observations - 1):
            for i in range(hidden_states):
                for j in range(hidden_states):
                    Fit = F[i, t]
                    aij = Transition[i, j]
                    bjt1 = Emission[j, Observations[t + 1]]
                    Bjt1 = B[j, t + 1]
                    NUM[i, j, t] = Fit * aij * bjt1 * Bjt1

        DEN = np.sum(NUM, axis=(0, 1))
        X = NUM / DEN


        G = np.zeros((hidden_states, observations))
        NUM = np.zeros((hidden_states, observations))


        for t in range(observations):
            for i in range(hidden_states):
                Fit = F[i, t]
                Bit = B[i, t]
                NUM[i, t] = Fit * Bit

        DEN = np.sum(NUM, axis=0)
        G = NUM / DEN

        Transition = np.sum(
            X, axis=2) / np.sum(
                G[:, :observations - 1], axis=1)[..., np.newaxis]

        DEN = np.sum(G, axis=1)
        NUM = np.zeros((hidden_states, output_states))
        for k in range(output_states):
            NUM[:, k] = np.sum(G[:, Observations == k], axis=1)
        Emission = NUM / DEN[..., np.newaxis]

        if np.all(
            np.isclose(
                Transition, transition_prev)) or np.all(
                    np.isclose(Emission, emission_prev)):
            return Transition, Emission

        transition_prev = np.copy(Transition)
        emission_prev = np.copy(Emission)

    return Transition, Emission
