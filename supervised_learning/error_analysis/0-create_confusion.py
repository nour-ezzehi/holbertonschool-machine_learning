#!/usr/bin/env python3
""" 0. Create Confusion """

import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates a confusion matrix """

    classes = labels.shape[1]
    confusion_matrix = np.zeros((classes, classes), dtype=float)
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)
    for true, pred in zip(true_labels, predicted_labels):
        confusion_matrix[true, pred] += 1

    return confusion_matrix
