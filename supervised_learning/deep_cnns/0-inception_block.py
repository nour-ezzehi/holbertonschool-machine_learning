#!/usr/bin/env python3
""" 0. Inception Block """


from tensorflow import keras as K


def inception_block(A_prev, filters):
    """ builds an inception block """

    init = K.initializers.he_normal()

    F1, F3R, F3, F5R, F5, FPP = filters
    c_F1 = K.layers.Conv2D(F1, kernel_size=1, padding="same",
                           activation="relu", kernel_initializer=init)(A_prev)

    c_F3R = K.layers.Conv2D(F3R, kernel_size=1, padding="same",
                            activation="relu", kernel_initializer=init)(A_prev)

    c_F3 = K.layers.Conv2D(F3, kernel_size=3, padding="same",
                           activation="relu", kernel_initializer=init)(c_F3R)

    c_F5R = K.layers.Conv2D(F5R, kernel_size=1, padding="same",
                            activation="relu", kernel_initializer=init)(A_prev)

    c_F5 = K.layers.Conv2D(F5, kernel_size=5, padding="same",
                           activation="relu", kernel_initializer=init)(c_F5R)

    pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                 padding="same")(A_prev)

    c_FPP = K.layers.Conv2D(FPP, kernel_size=1, padding="same",
                            activation="relu", kernel_initializer=init)(pool)

    output = K.layers.concatenate([c_F1, c_F3, c_F5, c_FPP])

    return output
