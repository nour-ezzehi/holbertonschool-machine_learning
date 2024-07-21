#!/usr/bin/env python3
""" 4. Inverse """
adjugate = __import__('3-adjugate').adjugate
determinant = __import__('0-determinant').determinant


def inverse(matrix):
    """ calculates the inverse of a matrix """
    adjugate_matrix = adjugate(matrix)
    det_value = determinant(matrix)

    if det_value == 0:
        return None
    inverse_matrix = []

    for i in range(len(adjugate_matrix)):
        row = []
        for j in range(len(adjugate_matrix[0])):
            row.append(adjugate_matrix[i][j] / det_value)
        inverse_matrix.append(row)
    return inverse_matrix
