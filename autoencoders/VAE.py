import keras
import tensorflow as tf
from keras import layers, Input, Model
from functools import reduce
from keras.utils.vis_utils import plot_model
from .sampling import Sampling
from .internal import VAEBallTrackInternal, VAEBallTrackNoPaddleInternal, VAEInternal, VAEInternalPerceptual
from .coord_conv import CoordinateChannel2D


class VAE():
    def __init__(self, layers, input_shape, latent_size, kernel_size=3, filters=8, name='Autoencoder'):
        self.layers = layers
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.pooling_factor = (2, 2)
        self.kernel_size = (kernel_size, kernel_size)
        self.initial_filters = filters
        self.final_filters = 0
        self.flat_size = 0
        self.reshaping_shape = (1,)
        self.input_size = reduce((lambda x, y: x*y), input_shape)

        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()

        self.model = VAEInternal(self.encoder, self.decoder, name=name)

        batch_input = list(input_shape)
        batch_input.insert(0, None)
        batch_input = tuple(batch_input)
        self.model.build(batch_input)
        self.model.compile(optimizer=keras.optimizers.Adam())

    def create_encoder(self):
        filters = self.initial_filters
        input_img = Input(shape=self.input_shape, name='input_encoder')
        x = input_img
        for i in range(self.layers):
            x = layers.Conv2D(filters, self.kernel_size, activation='relu',
                              padding='same', name=f'conv{i+1}_enc')(x)
            x = layers.MaxPooling2D(
                self.pooling_factor, padding='same', name=f'maxpool{i+1}')(x)
            filters *= 2

        flat = layers.Flatten(name='flatten')(x)
        bn = layers.Dense(self.latent_size, name='bottleneck')(flat)

        z_mean = layers.Dense(self.latent_size, name="z_mean")(bn)
        z_log_var = layers.Dense(self.latent_size, name="z_log_var")(bn)
        z = Sampling()([z_mean, z_log_var])

        # Saving values for the decoder
        self.reshaping_shape = x.shape[1:]
        self.final_filters = filters / 2
        self.flat_size = flat.shape[1]

        return Model(input_img, [z_mean, z_log_var, z], name='Encoder')

    def create_decoder(self):
        filters = self.final_filters

        input_decoder = Input(shape=(self.latent_size,), name='input_decoder')

        dec = layers.Dense(self.flat_size, name="decoding")(input_decoder)
        reshaped = layers.Reshape(self.reshaping_shape, name='reshape')(dec)
        x = reshaped

        for i in range(self.layers):
            x = layers.Conv2D(filters, self.kernel_size, activation='relu',
                              padding='same', name=f'conv{self.layers-i}_dec')(x)
            x = layers.UpSampling2D(
                self.pooling_factor, name=f'upsamp{self.layers-i}')(x)
            filters /= 2

        # Calculate the kernel size for the last layer, in order for it to have the same
        # width and height as the input image
        kernel_y = x.shape[1] - self.input_shape[0] + 1
        kernel_x = x.shape[2] - self.input_shape[1] + 1

        decoded = layers.Conv2D(
            self.input_shape[-1], (kernel_y, kernel_x), activation='sigmoid', padding='valid', name='output')(x)

        return Model(input_decoder, decoded, name="Decoder")

    def train(self, train_gen, val_gen, epochs, batch_size=32, callbacks=[]):
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=val_gen,
            callbacks=callbacks)

        return history

    # Train by inputting tensors and not generators
    def train_primitive(self, train, val, epochs, batch_size=32, callbacks=[]):
        history = self.model.fit(
            x=train, y=train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(val, val),
            callbacks=callbacks)

        return history

    def predict(self, imgs):
        return self.model.predict(imgs)

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def evaluate(self, data):
        rec = self.model.predict(data)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, rec), axis=(1, 2)
            )
        )

        return reconstruction_loss.numpy()


class VAEPerceptual(VAE):
    def __init__(self, layers, input_shape, latent_size, kernel_size, filters, name):
        super().__init__(layers, input_shape, latent_size,
                         kernel_size=kernel_size, filters=filters, name=name)

        self.model = VAEInternalPerceptual(
            self.encoder, self.decoder, name=name)

        batch_input = list(input_shape)
        batch_input.insert(0, None)
        batch_input = tuple(batch_input)
        self.model.build(batch_input)
        self.model.compile(optimizer=keras.optimizers.Adam())

    def summary(self):
        self.model.summary_perceptual()
        super().summary()


class VAEBallTrack(VAE):
    def __init__(self, layers, input_shape, latent_size, kernel_size, name):
        super().__init__(layers, input_shape, latent_size,
                         kernel_size=kernel_size, filters=filters, name=name)

        self.model = VAEBallTrackInternal(
            self.encoder, self.decoder, name=name)

        batch_input = list(input_shape)
        batch_input.insert(0, None)
        batch_input = tuple(batch_input)
        self.model.build(batch_input)
        self.model.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)

    def summary(self):
        super().summary()


class VAEBallTrackNoPaddle(VAE):
    def __init__(self, layers, input_shape, latent_size, kernel_size, filters, name):
        super().__init__(layers, input_shape, latent_size,
                         kernel_size=kernel_size, filters=filters, name=name)

        self.model = VAEBallTrackNoPaddleInternal(
            self.encoder, self.decoder, name=name)

        batch_input = list(input_shape)
        batch_input.insert(0, None)
        batch_input = tuple(batch_input)
        self.model.build(batch_input)
        self.model.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)

    def summary(self):
        super().summary()

    def train(self, train_gen, val_gen, epochs, batch_size=32, callbacks=[]):
        self.model.setup_coordinate_array(batch_size)
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=val_gen,
            callbacks=callbacks)

        return history

    # Train by inputting tensors and not generators
    def train_primitive(self, train, val, epochs, batch_size=32, callbacks=[]):
        self.model.setup_coordinate_array(batch_size)
        history = self.model.fit(
            x=train, y=train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(val, val),
            callbacks=callbacks)

        return history


class VAECoordConv(VAE):
    def __init__(self, layers, input_shape, latent_size, kernel_size, filters, name):
        super().__init__(layers, input_shape, latent_size,
                         kernel_size=kernel_size, filters=filters, name=name)

    def create_encoder(self):
        filters = self.initial_filters
        input_img = Input(shape=self.input_shape, name='input_encoder')
        x = input_img

        x = CoordinateChannel2D()(x)

        for i in range(self.layers):
            x = layers.Conv2D(filters, self.kernel_size, activation='relu',
                              padding='same', name=f'conv{i+1}_enc')(x)
            x = layers.MaxPooling2D(
                self.pooling_factor, padding='same', name=f'maxpool{i+1}')(x)
            filters *= 2

        flat = layers.Flatten(name='flatten')(x)
        bn = layers.Dense(self.latent_size, name='bottleneck')(flat)

        z_mean = layers.Dense(self.latent_size, name="z_mean")(bn)
        z_log_var = layers.Dense(self.latent_size, name="z_log_var")(bn)
        z = Sampling()([z_mean, z_log_var])

        # Saving values for the decoder
        self.reshaping_shape = x.shape[1:]
        self.final_filters = filters / 2
        self.flat_size = flat.shape[1]

        return Model(input_img, [z_mean, z_log_var, z], name='Encoder')

    def create_decoder(self):
        filters = self.final_filters

        input_decoder = Input(shape=(self.latent_size,), name='input_decoder')

        dec = layers.Dense(self.flat_size, name="decoding")(input_decoder)
        reshaped = layers.Reshape(self.reshaping_shape, name='reshape')(dec)
        x = reshaped

        for i in range(self.layers):
            #x = CoordinateChannel2D()(x)
            x = layers.Conv2D(filters, self.kernel_size, activation='relu',
                              padding='same', name=f'conv{self.layers-i}_dec')(x)
            x = layers.UpSampling2D(
                self.pooling_factor, name=f'upsamp{self.layers-i}')(x)
            filters /= 2

        # Calculate the kernel size for the last layer, in order for it to have the same
        # width and height as the input image
        kernel_y = x.shape[1] - self.input_shape[0] + 1
        kernel_x = x.shape[2] - self.input_shape[1] + 1

        x = CoordinateChannel2D()(x)
        # Add coordinate filter in last conv layer
        decoded = layers.Conv2D(
            self.input_shape[-1], (kernel_y, kernel_x), activation='sigmoid', padding='valid', name='output')(x)

        return Model(input_decoder, decoded, name="Decoder")
