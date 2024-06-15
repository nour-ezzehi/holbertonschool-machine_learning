#!/usr/bin/env python3
"""  1. Inception Network """


from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ builds the inception network """

    init = K.initializers.HeNormal()
    inputs = K.Input((224, 224, 3))
    c1 = K.layers.Conv2D(
        64,
        7,
        2,
        activation='relu',
        padding='same',
        kernel_initializer=init)(inputs)

    Mpool1 = K.layers.MaxPool2D((3, 3), 2, padding='same')(c1)
    c2 = K.layers.Conv2D(
        64,
        1,
        1,
        activation='relu',
        padding='same',
        kernel_initializer=init)(Mpool1)
    c3 = K.layers.Conv2D(
        192,
        3,
        1,
        activation='relu',
        padding='same',
        kernel_initializer=init)(c2)
    M2pool = K.layers.MaxPool2D((3, 3), 2, padding='same')(c3)
    i1 = inception_block(M2pool, [64, 96, 128, 16, 32, 32])
    i2 = inception_block(i1, [128, 128, 192, 32, 96, 64])
    M3pool = K.layers.MaxPool2D((3, 3), 2, padding='same')(i2)
    i4 = inception_block(M3pool, [192, 96, 208, 16, 48, 64])
    i5 = inception_block(i4, [160, 112, 224, 24, 64, 64])
    i6 = inception_block(i5, [128, 128, 256, 24, 64, 64])
    i7 = inception_block(i6, [112, 144, 288, 32, 64, 64])
    i8 = inception_block(i7, [256, 160, 320, 32, 128, 128])
    M4pool = K.layers.MaxPool2D((3, 3), 2, padding='same')(i8)
    i9 = inception_block(M4pool, [256, 160, 320, 32, 128, 128])
    i10 = inception_block(i9, [384, 192, 384, 48, 128, 128])

    avg_pool = K.layers.AveragePooling2D((7, 7), 1)(i10)
    drop = K.layers.Dropout(0.4)(avg_pool)
    linear = K.layers.Dense(1000, activation='softmax',
                            kernel_initializer=init)(drop)

    model = K.Model(inputs=inputs, outputs=linear)

    return model
