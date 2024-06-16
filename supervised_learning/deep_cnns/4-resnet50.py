#!/usr/bin/env python3
""" 4. ResNet-50 """


from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ builds the ResNet-50 architecture """

    img_input = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()
    L1 = K.layers.Conv2D(filters=64,
                         kernel_size=(7, 7),
                         padding='same',
                         strides=(2, 2),
                         kernel_initializer=init)(img_input)
    L1 = K.layers.BatchNormalization(axis=3)(L1)
    L1 = K.layers.Activation('relu')(L1)
    L1 = K.layers.MaxPool2D((3, 3), (2, 2), padding='same')(L1)
    L1 = projection_block(L1, [64, 64, 256], 1)
    L1 = identity_block(L1, [64, 64, 256])
    L1 = identity_block(L1, [64, 64, 256])
    L1 = projection_block(L1, [128, 128, 512])
    L1 = identity_block(L1, [128, 128, 512])
    L1 = identity_block(L1, [128, 128, 512])
    L1 = identity_block(L1, [128, 128, 512])
    L1 = projection_block(L1, [256, 256, 1024])
    L1 = identity_block(L1, [256, 256, 1024])
    L1 = identity_block(L1, [256, 256, 1024])
    L1 = identity_block(L1, [256, 256, 1024])
    L1 = identity_block(L1, [256, 256, 1024])
    L1 = identity_block(L1, [256, 256, 1024])
    L1 = projection_block(L1, [512, 512, 2048])
    L1 = identity_block(L1, [512, 512, 2048])
    L1 = identity_block(L1, [512, 512, 2048])
    L1 = K.layers.AvgPool2D((7, 7), padding='same')(L1)
    L1 = K.layers.Dense(1000, activation='softmax')(L1)
    model = K.Model(img_input, L1)

    return model
