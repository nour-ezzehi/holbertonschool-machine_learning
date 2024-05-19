#!/usr/bin/env python3
""" 4. Moving Average
"""


def moving_average(data, beta):
    """ calculates the weighted moving average of a data set
    """
    v = 0
    MA = list()
    for i in range(len(data)):
        v = (beta * v) + ((1 - beta) * data[i])
        bias_correction = 1 - beta ** (i + 1)
        new_v = v / bias_correction
        MA.append(new_v)

    return MA
