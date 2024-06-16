#!/usr/bin/env python3


""" 2. Identity Block """


from tensorflow import keras as K


def identity_block(A_prev, filters):
    """ builds an identity block """

    init = K.initializers.he_normal(seed=0)
    F11, F3, F12 = filters

    c_F11 = K.layers.Conv2D(F11, kernel_size=1, padding="same",
                            kernel_initializer=init)(A_prev)
    norm_F11 = K.layers.BatchNormalization()(c_F11)
    act_F11 = K.layers.Activation("relu")(norm_F11)

    c_F3 = K.layers.Conv2D(F3, kernel_size=3, padding="same",
                           kernel_initializer=init)(act_F11)
    norm_F3 = K.layers.BatchNormalization()(c_F3)
    act_F3 = K.layers.Activation("relu")(norm_F3)

    c_F12 = K.layers.Conv2D(F12, kernel_size=1, padding="same",
                            kernel_initializer=init)(act_F3)
    norm_F12 = K.layers.BatchNormalization()(c_F12)

    X = K.layers.Add()([norm_F12, A_prev])

    output = K.layers.Activation("relu")(X)

    return output
