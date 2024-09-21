#!/usr/bin/env python3
""" 0. "Vanilla" Autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ creates an autoencoder """
    input_img = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(hidden_layers[0], activation='relu')
    (input_img)

    for hl in hidden_layers[1:]:
        encoded = keras.layers.Dense(hl, activation='relu')(encoded)

    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.Model(input_img, latent)

    input_latent = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(hidden_layers[-1], activation='relu')
    (input_latent)

    for hl in hidden_layers[-2::-1]:
        decoded = keras.layers.Dense(hl, activation='relu')(decoded)

    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(input_latent, decoded)

    auto = keras.Model(input_img, decoder(encoder(input_img)))

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
