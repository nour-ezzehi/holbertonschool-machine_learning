#!/usr/bin/env python3
"""2. Size Me Please"""


def matrix_shape(matrix):
    """calculates the shape of a matrix without using recursion"""
    shape = list()
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
