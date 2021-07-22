from keras import layers, Input, Model
from functools import reduce
import tensorflow as tf
import keras
from .coord_conv import CoordinateChannel2D
from .ball_track_metric import ball_track_metric


class CAE():
    def __init__(self, layers, input_shape, latent_size, kernel_size=3, filters=16, name='Autoencoder'):
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

        input_img = Input(shape=self.input_shape, name='input_encoder')

        encoded_img = self.encoder(input_img)
        decoded_img = self.decoder(encoded_img)

        self.model = Model(input_img, decoded_img, name=name)
        self.model.compile(
            optimizer='adam', loss='binary_crossentropy')

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
        encoded = layers.Dense(self.latent_size, name='bottleneck')(flat)

        # Saving values for the decoder
        self.reshaping_shape = x.shape[1:]
        self.final_filters = filters / 2
        self.flat_size = flat.shape[1]

        return Model(input_img, encoded, name='Encoder')

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
            1, (kernel_y, kernel_x), activation='sigmoid', padding='valid', name='output')(x)

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

        ball_loss = ball_track_metric(rec, data)

        return {"reconstruction": reconstruction_loss.numpy(), "ball_track": ball_loss.numpy()}


class CAECoordConv(CAE):
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
        encoded = layers.Dense(self.latent_size, name='bottleneck')(flat)

        # Saving values for the decoder
        self.final_filters = filters / 2
        self.flat_size = flat.shape[1]
        # Reshaping shape witll have the same size as original image, and the amount of filters is calculating by
        # dividing the flat size by the size of the input square, in order to have the same size as the flat size.
        self.reshaping_shape = (
            *self.input_shape[:-1], int((self.flat_size) / (self.input_shape[0] * self.input_shape[1])))

        return Model(input_img, encoded, name='Encoder')

    def create_decoder(self):
        filters = self.final_filters

        input_decoder = Input(shape=(self.latent_size,), name='input_decoder')
        x = input_decoder

        dec = layers.Dense(self.flat_size, name="decoding")(x)
        reshaped = layers.Reshape(self.reshaping_shape, name='reshape')(dec)
        x = reshaped
        x = CoordinateChannel2D()(x)

        for i in range(self.layers):
            # Uber paper has conv layers of kernel 1x1 por decoder
            x = layers.Conv2D(filters, (1, 1), activation='relu',
                              padding='same', name=f'conv{self.layers-i}_dec')(x)
            # x = layers.UpSampling2D(
            #    self.pooling_factor, name=f'upsamp{self.layers-i}')(x)
            filters /= 2

        # Calculate the kernel size for the last layer, in order for it to have the same
        # width and height as the input image
        kernel_y = x.shape[1] - self.input_shape[0] + 1
        kernel_x = x.shape[2] - self.input_shape[1] + 1

        decoded = layers.Conv2D(
            1, (kernel_y, kernel_x), activation='sigmoid', padding='valid', name='output')(x)

        return Model(input_decoder, decoded, name="Decoder")
