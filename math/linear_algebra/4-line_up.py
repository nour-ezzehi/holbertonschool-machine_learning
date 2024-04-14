#!/usr/bin/env python3
"""4. Line Up"""


def add_arrays(arr1, arr2):
    """add two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    return [x + y for x, y in zip(arr1, arr2)]
