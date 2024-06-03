#!/usr/bin/env python3
""" 0. Sequential """

import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ builds a neural network with the Keras library """

    model = k.Sequential()
    regularizer = k.regularizers.l2(lambtha)

    for i, (layer_size, activation) in enumerate(zip(layers, activations)):
        if i == 0:
            model.add(
                k.layers.Dense(
                    layer_size,
                    activation=activation,
                    input_shape=(
                        nx,
                    ),
                    kernel_regularizer=regularizer))
        else:
            model.add(
                k.layers.Dense(
                    layer_size,
                    activation=activation,
                    kernel_regularizer=regularizer))

        if i < len(layers) - 1:
            model.add(k.layers.Dropout(1 - keep_prob))

    return model
