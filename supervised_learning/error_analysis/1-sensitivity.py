#!/usr/bin/env python3
""" Sensitivity """


import numpy as np


def sensitivity(confusion):
    """ calculates the sensitivity for each class in a confusion matrix """
    classes = confusion.shape[1]
    sensitivities = np.zeros(classes)

    for i in range(classes):
        true_positives = confusion[i, i]
        false_negatives = np.sum(confusion[i, :]) - true_positives
        sensitivities[i] = true_positives / (true_positives + false_negatives)

    return sensitivities
