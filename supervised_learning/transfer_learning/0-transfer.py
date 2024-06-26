#!/usr/bin/env python3
""" 0. Transfer Knowledge """


from tensorflow import keras as K


def preprocess_data(X, Y):
    """ pre-processes the data for the model """
    X_p = K.applications.vgg16.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    """ Trains a CNN based on VGG16 model to classify the CIFAR 10 dataset """
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    input = K.Input(shape=(32, 32, 3))

    scaled_up_layer = K.layers.Lambda(
        lambda i: K.backend.resize_images(
            i,
            height_factor=1.5,
            width_factor=1.5,
            data_format='channels_last'))(input)

    base_model = K.applications.VGG16(weights='imagenet',
                                      include_top=False,
                                      input_tensor=scaled_up_layer,
                                      input_shape=(48, 48, 3))

    last = base_model.get_layer('block3_pool').output

    base_model.trainable = False

    layer = K.layers.GlobalAveragePooling2D()(last)
    layer = K.layers.BatchNormalization()(layer)
    layer = K.layers.Dense(units=256, activation='relu')(layer)
    layer = K.layers.Dropout(rate=0.6)(layer)

    classes = 10
    output = K.layers.Dense(units=classes, activation='softmax')(layer)
    model = K.Model(input, output)
    Adam = K.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=128,
                        validation_data=(x_test, y_test),
                        epochs=20)

    model.save('cifar10.h5')
