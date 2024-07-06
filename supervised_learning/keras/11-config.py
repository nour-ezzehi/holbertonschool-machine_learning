#!/usr/bin/env python3
""" 11. Save and Load Configuration """


import tensorflow.keras as K


def save_config(network, filename):
    """ saves a model’s configuration in JSON format """
    network_json = network.to_json()
    with open(filename, "w") as f:
        f.write(network_json)


def load_config(filename):
    """ loads a model’s with specific configuration"""
    with open(filename, "r") as f:
        loaded_network_json = f.read()

    return K.models.model_from_json(loaded_network_json)
