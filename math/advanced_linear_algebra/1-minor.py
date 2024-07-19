#!/usr/bin/env python3
""" 1. Minor """


def determinant(matrix):
    """ calculates the determinant of a matrix """
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        det = ((matrix[0][0] * matrix[1][1])
               - (matrix[0][1] * matrix[1][0]))
        return det

    det = 0
    for i, j in enumerate(matrix[0]):
        row = [r for r in matrix[1:]]
        temp = []
        for r in row:
            a = []
            for c in range(len(matrix)):
                if c != i:
                    a.append(r[c])
            temp.append(a)
        det += j * (-1) ** i * determinant(temp)
    return det


def minor(matrix):
    """ calculates the minor matrix of a matrix """

    if not isinstance(matrix, list) or matrix == []:
        raise TypeError('matrix must be a list of lists')
    if any(not isinstance(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == 1:
        return [[1]]

    minor_matrix = []
    for x in range(len(matrix)):
        t = []
        for y in range(len(matrix[0])):
            s = []
            for row in (matrix[:x] + matrix[x + 1:]):
                s.append(row[:y] + row[y + 1:])
            t.append(determinant(s))
        minor_matrix.append(t)
    return minor_matrix
