#!/usr/bin/env python3
""" Specifity """


import numpy as np


def specificity(confusion):
    """ calculates the specificity for each class in a confusion matrix """
    classes = confusion.shape[1]
    specificities = np.zeros(classes)

    for i in range(classes):
        predictions = np.sum(confusion)
        row_sum = np.sum(confusion[i, :])
        column_sum = np.sum(confusion[:, i])
        true_negatives = predictions - row_sum - column_sum + confusion[i, i]

        false_positives = np.sum(confusion[:, i]) - confusion[i, i]
        specificities[i] = true_negatives / (true_negatives + false_positives)

    return specificities
