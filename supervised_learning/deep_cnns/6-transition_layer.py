#!/usr/bin/env python3
""" 6. Transition Layer """


from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """ builds a transition layer """
    init = K.initializers.he_normal(seed=0)
    n_number = int(nb_filters * compression)

    norm0 = K.layers.BatchNormalization()(X)
    act0 = K.layers.Activation("relu")(norm0)
    conv = K.layers.Conv2D(filters=int(n_number),
                           kernel_size=(1, 1),
                           padding="same",
                           strides=(1, 1),
                           kernel_initializer=init)(act0)

    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2),
                                         padding="same")(conv)

    return avg_pool, n_number
