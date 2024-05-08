#!/usr/bin/env python3
"""DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """ defines a deep neural network performing binary classification """

    def __init__(self, nx, layers):
        """constructor for DeepNeuralNetwork"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                he_initialized_weights = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                he_initialized_weights = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.weights['W' + str(i + 1)] = he_initialized_weights
            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """getter for L"""
        return self.__L

    @property
    def cache(self):
        """getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """getter for weights"""
        return self.__weights
