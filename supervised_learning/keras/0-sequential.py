#!/usr/bin/env python3
""" 0. Sequential """

import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ builds a neural network with the Keras library """

    model = k.Sequential()
    regularizer = k.regularizers.l2(lambtha)

    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(
                k.layers.Dense(
                    nodes,
                    activation=activations[i],
                    kernel_regularizer=regularizer,
                    input_shape=(
                        nx,
                    )))
        else:
            model.add(
                k.layers.Dense(
                    nodes,
                    activation=activations[i],
                    kernel_regularizer=regularizer))

        if keep_prob < 1:
            model.add(k.layers.Dropout(1 - keep_prob))

    return model
