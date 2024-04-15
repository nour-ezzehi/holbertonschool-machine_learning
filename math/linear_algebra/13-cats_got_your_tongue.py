#!/usr/bin/env python3
"""13. Cat's Got Your Tongue"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenate two matrices"""
    return np.concatenate((mat1, mat2), axis=axis)
