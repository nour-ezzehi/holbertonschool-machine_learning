#!/usr/bin/env python3
""" 9. Save and Load Model """


import tensorflow.keras as K


def save_model(network, filename):
    """ Saves a model """

    network.save(filename)


def load_model(filename):
    """ Loads a model """

    return K.models.load_model(filename)
