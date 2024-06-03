#!/usr/bin/env python3
""" 1. Input """

import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ builds a neural network with the Keras library """

    inputs = k.Input(shape=(nx,))
    x = inputs
    regularizer = k.regularizers.l2(lambtha)

    for i, (layer_size, activation) in enumerate(zip(layers, activations)):
        x = k.layers.Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=regularizer)(x)
        if i < len(layers) - 1:
            x = k.layers.Dropout(1 - keep_prob)(x)

    model = k.Model(inputs=inputs, outputs=x)

    return model
