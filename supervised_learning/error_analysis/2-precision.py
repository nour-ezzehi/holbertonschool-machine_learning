#!/usr/bin/env python3
""" Precision"""


import numpy as np


def precision(confusion):
    """ calculates the precision for each class in a confusion matrix """
    classes = confusion.shape[1]
    precisions = np.zeros(classes)

    for i in range(classes):
        true_positives = confusion[i, i]
        false_positives = np.sum(confusion[:, i]) - true_positives
        precisions[i] = true_positives / (true_positives + false_positives)

    return precisions
