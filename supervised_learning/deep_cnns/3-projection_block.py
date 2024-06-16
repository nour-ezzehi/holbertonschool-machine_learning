#!/usr/bin/env python3
""" 3. Projection Block """


from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """ builds a projection block """

    init = K.initializers.he_normal(seed=0)
    F11, F3, F12 = filters

    c_F11 = K.layers.Conv2D(F11, kernel_size=1, padding="same", strides=s,
                            kernel_initializer=init)(A_prev)
    norm_F11 = K.layers.BatchNormalization()(c_F11)
    act_F11 = K.layers.Activation("relu")(norm_F11)

    c_F3 = K.layers.Conv2D(F3, kernel_size=3, padding="same", strides=1,
                           kernel_initializer=init)(act_F11)
    norm_F3 = K.layers.BatchNormalization()(c_F3)
    act_F3 = K.layers.Activation("relu")(norm_F3)
    c_F12 = K.layers.Conv2D(F12, kernel_size=1, padding="same", strides=1,
                            kernel_initializer=init)(act_F3)
    norm_F12 = K.layers.BatchNormalization()(c_F12)

    shortcut = K.layers.Conv2D(F12, kernel_size=1, padding="same", strides=s,
                               kernel_initializer=init)(A_prev)
    shortcut = K.layers.BatchNormalization()(shortcut)

    X = K.layers.Add()([norm_F12, shortcut])

    output = K.layers.Activation("relu")(X)

    return output
