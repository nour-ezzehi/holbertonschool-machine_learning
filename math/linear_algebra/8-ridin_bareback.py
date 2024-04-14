#!/usr/bin/env python3
"""8. Ridin’ Bareback"""


def mat_mul(mat1, mat2):
    """performs matrix multiplication"""
    new_matrix = []
    for i in range(len(mat1)):
        new_matrix.append([])
        for j in range(len(mat2[0])):
            sum = 0
            for k in range(len(mat1[0])):
                sum += mat1[i][k] * mat2[k][j]
            new_matrix[i].append(sum)

    return new_matrix
