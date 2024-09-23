#!/usr/bin/env python3
"""This module contains a simple GAN class that inherits from keras.Model"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):

    def __init__(
            self,
            generator,
            discriminator,
            latent_generator,
            real_examples,
            batch_size=200,
            disc_iter=2,
            learning_rate=.005):
        # run the __init__ of keras.Model first.
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        # standard value, but can be changed if necessary
        self.beta_1 = .5
        # standard value, but can be changed if necessary
        self.beta_2 = .9
        # Define the generator loss and optimizer:
        self.generator.loss = lambda x: - tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(
            optimizer=generator.optimizer,
            loss=generator.loss)

        # Define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: (
            -tf.math.reduce_mean(x) + tf.math.reduce_mean(y))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss)
    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()
    def train_step(self, useless_argument):
        """ train the GAN for one step """

        for _ in range(self.disc_iter):
            real_sample = self.get_real_sample()
            fake_sample = self.get_fake_sample()

            with tf.GradientTape() as tape:
                real_disc_output = self.discriminator(real_sample)
                fake_disc_output = self.discriminator(fake_sample)

                discr_loss = self.discriminator.loss(x=real_disc_output,
                                                     y=fake_disc_output)

            disc_gradients = tape.gradient(
                discr_loss, self.discriminator.trainable_variables)

            self.discriminator.optimizer.apply_gradients(zip(
                disc_gradients,
                self.discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample()
            gen_out = self.discriminator(fake_sample, training=True)

            gen_loss = self.generator.loss(gen_out)

        gen_gradients = tape.gradient(gen_loss,
                                      self.generator.trainable_variables)

        self.generator.optimizer.apply_gradients(zip(
            gen_gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
