#!/usr/bin/env python3
""" 0. Determinant """


def determinant(matrix):
    """ calculates the determinant of a matrix """
    if not all(isinstance(row, list)
               for row in matrix) or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    if n == 0:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) != n and n != 1:
            raise ValueError("matrix must be a square matrix")

    if matrix == [[]]:
        return 1

    mat = [row[:] for row in matrix]

    det = 1
    for i in range(n):
        pivot = i
        for j in range(i + 1, n):
            if abs(mat[j][i]) > abs(mat[pivot][i]):
                pivot = j
        if mat[pivot][i] == 0:
            return 0

        mat[i], mat[pivot] = mat[pivot], mat[i]
        if i != pivot:
            det *= -1

        det *= mat[i][i]

        for j in range(i + 1, n):
            factor = mat[j][i] / mat[i][i]
            for k in range(i, n):
                mat[j][k] -= factor * mat[i][k]

    return int(det)
