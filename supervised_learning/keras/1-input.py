#!/usr/bin/env python3
""" 1. Input """

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ builds a neural network with the Keras library """

    inputs = K.Input(shape=(nx,))
    x = inputs
    regularizer = K.regularizers.l2(lambtha)

    for i, (layer_size, activation) in enumerate(zip(layers, activations)):
        x = K.layers.Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=regularizer)(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)

    return model
