#!/usr/bin/env python3
"""3. Flip Me Over"""


def matrix_transpose(matrix):
    """
    returns the transpose of a 2D matrix
      """
    new_matrix = []
    for i in range(len(matrix[0])):
        new_matrix.append([])
        for j in range(len(matrix)):
            new_matrix[i].append(matrix[j][i])
    return new_matrix
