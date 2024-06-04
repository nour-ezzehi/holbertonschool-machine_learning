#!/usr/bin/env python3
""" 4. Train """


import tensorflow.keras as K


def train_model(
        network, data, labels, batch_size, epochs, validation_data=None,
        verbose=True, shuffle=False):
    """ update the function def train_model to also analyze validaiton data """

    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data)
