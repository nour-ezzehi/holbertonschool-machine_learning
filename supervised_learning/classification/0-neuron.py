#!/usr/bin/env python3
"""binary classification"""

import numpy as np


class Neuron():
    """neuron class"""
    def __init__(self, nx):
        """constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
