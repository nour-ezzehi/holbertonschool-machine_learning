#!/usr/bin/env python3
""" 1. Normalize """


def normalize(X, m, s):
    """ normalizes (standardizes) a matrix
    """

    return (X - m) / s
