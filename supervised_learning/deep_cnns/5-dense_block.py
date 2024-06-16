#!/usr/bin/env python3
""" 5. Dense Block """


from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ builds a dense block """
    he_normal = K.initializers.HeNormal(seed=0)

    for i in range(layers):
        X1 = K.layers.BatchNormalization(axis=-1)(X)
        X1 = K.layers.ReLU()(X1)
        X1 = K.layers.Conv2D(4 * growth_rate, (1, 1), padding='same',
                             kernel_initializer=he_normal)(X1)

        X1 = K.layers.BatchNormalization(axis=-1)(X1)
        X1 = K.layers.ReLU()(X1)
        X1 = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                             kernel_initializer=he_normal)(X1)

        X = K.layers.Concatenate(axis=-1)([X, X1])

        nb_filters += growth_rate

    return X, nb_filters
