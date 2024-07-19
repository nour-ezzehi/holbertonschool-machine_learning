#!/usr/bin/env python3
""" 2. Cofactor """
minor = __import__('1-minor').minor


def cofactor(matrix):
    """ calculates the cofactor matrix of a matrix """
    minor_matrix = minor(matrix)
    n = len(matrix)
    cof_matrix = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            sign = (-1) ** (i + j)
            cof_matrix[i][j] = sign * minor_matrix[i][j]

    return cof_matrix
