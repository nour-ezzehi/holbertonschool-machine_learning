#!/usr/bin/env python3
""" 10. Save and Load Weights """


import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ saves a model’s weights"""

    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """ loads a model’s weights """

    network.load_weights(filename)
