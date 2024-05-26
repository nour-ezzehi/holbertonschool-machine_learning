#!/usr/bin/env python3
""" F1 score """


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ calculates the F1 score of a confusion matrix """
    classes = confusion.shape[1]
    f1_scores = np.zeros(classes)
    sensitivities = sensitivity(confusion)
    precisions = precision(confusion)

    for i in range(classes):
        f1_scores[i] = 2 * (precisions[i] * sensitivities[i]) / \
            (precisions[i] + sensitivities[i])

    return f1_scores
