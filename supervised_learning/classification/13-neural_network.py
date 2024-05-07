#!/usr/bin/env python3
"""NeuralNetwork"""
import numpy as np


class NeuralNetwork:
    """a neural network"""

    def __init__(self, nx, nodes):
        """Constructor method"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b1 = np.zeros((nodes, 1))
        self.__b2 = 0
        self.__A1 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ getter function for Weights vector """
        return self.__W1

    @property
    def b1(self):
        """ getter function for bias """
        return self.__b1

    @property
    def A1(self):
        """ getter function for activation function output """
        return self.__A1

    @property
    def W2(self):
        """ getter function for Weights vector """
        return self.__W2

    @property
    def b2(self):
        """ getter function for bias """
        return self.__b2

    @property
    def A2(self):
        """ getter function for activation function output """
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        cost = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = np.sum(cost)
        cost = - cost / A.shape[1]

        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        _, A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, cost


def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
    """Calculates one pass of gradient descent on the neuron"""
    m = Y.shape[1]

    dz2 = A2 - Y
    dw2 = np.matmul(A1, dz2.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
    dw1 = np.matmul(X, dz1.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    self.__W2 -= alpha * dw2.T
    self.__b2 -= alpha * db2
    self.__W1 -= alpha * dw1.T
    self.__b1 -= alpha * db1
