import math
import keras
import numpy as np
import tensorflow as tf
from .perceptual_networks import PerceptualNetwork
from .ball_finder import find_balls, find_balls_tf


class VAEInternal(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAEInternal, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Calculate reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            # Calculate KL loss and total loss
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        # Get and propagate gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        # Calculate reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            )
        )
        # Calculate KL loss and total loss
        kl_loss = -0.5 * (1 + z_log_var -
                          tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


class VAEInternalPerceptual(VAEInternal):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(encoder, decoder, **kwargs)
        self.perceptual_model = PerceptualNetwork(
            model="VGG19",
            layer_amount=6,
            # Ignore batch size (1) and channels (-1) and add three channels at the end
            input_shape=tuple(list(encoder.input_shape)[1:-1] + [3])
        )

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            rgb_reconstruction = tf.image.grayscale_to_rgb(reconstruction)
            rgb_data = tf.image.grayscale_to_rgb(data)

            perception_rec = self.perceptual_model.call(
                rgb_reconstruction)  # shape (batch, 51200)
            perception_orig = self.perceptual_model.call(rgb_data)

            # Variable already tensor?
            # perception_rec = tf.constant(perception_rec)
            # perception_orig = tf.constant(perception_orig)

            diff = perception_rec - perception_orig
            sq_diff = tf.square(diff)  # shape (batch, 51200)

            mean_sq_diff = tf.reduce_mean(sq_diff, axis=1)  # shape (batch)

            # Calculate reconstruction loss
            reconstruction_loss = tf.reduce_mean(mean_sq_diff)  # shape 1

            # Calculate KL loss and total loss
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        # Get and propagate gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        rgb_reconstruction = tf.image.grayscale_to_rgb(reconstruction)
        rgb_data = tf.image.grayscale_to_rgb(data)

        perception_rec = self.perceptual_model.call(
            rgb_reconstruction)  # shape (batch, 51200)
        perception_orig = self.perceptual_model.call(rgb_data)

        # Variable already tensor?
        # perception_rec = tf.constant(perception_rec)
        # perception_orig = tf.constant(perception_orig)

        diff = perception_rec - perception_orig
        sq_diff = tf.square(diff)  # shape (batch, 51200)

        mean_sq_diff = tf.reduce_mean(sq_diff, axis=1)  # shape (batch)

        # Calculate reconstruction loss
        reconstruction_loss = tf.reduce_mean(mean_sq_diff)  # shape 1
        # Calculate KL loss and total loss
        kl_loss = -0.5 * (1 + z_log_var -
                          tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def summary_perceptual(self):
        self.perceptual_model.summary()


class VAEBallTrackInternal(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAEBallTrackInternal, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.diagonal = math.sqrt(
            encoder.input_shape[1] ** 2 + encoder.input_shape[2] ** 2)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.ball_loss_tracker = keras.metrics.Mean(name="ball_loss")
        self.coord_array = self.build_coordinate_array(32, 40, 40)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.ball_loss_tracker
        ]

    def build_coordinate_array(self, batch_size, screen_width, screen_height):
        result = np.zeros((screen_height, screen_width, 2), np.float32)
        for x in range(screen_width):  # todo: look up numpy mesh to do this faster?
            for y in range(screen_height):
                result[y, x, 0] = x
                result[y, x, 1] = y
        # should be shape [batch_size, screen_height, screen_width, 2]
        result = np.stack([result]*batch_size, axis=0)
        return tf.constant(result)

    def calculate_ball_loss(self, original, reconstruction):
        original_np = original.numpy()
        reconstruction_np = reconstruction.numpy()

        batch = len(original_np)

        balls_positions_orig = []
        balls_positions_rec = []

        for dp in original_np:
            balls_positions_orig.append(find_balls(dp, 5, 34))

        for dp in reconstruction_np:
            balls_positions_rec.append(find_balls(dp, 5, 34))

        loss = 0.0

        for ball_poss_or, ball_poss_rec in zip(balls_positions_orig, balls_positions_rec):
            # In the case where no ball is present in the original, we penalize with the
            # maximum loss, which is the diagnoal of the screen for each ball in the reconstuction.
            if len(ball_poss_or) < 1:
                loss += self.diagonal * len(ball_poss_rec)
            # If there is no ball in the reconstruction, but there is a ball in the original,
            # apply a diagonal loss times a factor
            elif len(ball_poss_rec) < 1:
                loss += self.diagonal * 4
            else:
                # The original only has one ball, so we retrieve that ball
                ball_orig = ball_poss_or[0]

                # We now calculate the euclidean distance of each ball to the original ball and sum
                # them to calculate the final loss
                loss += sum([math.sqrt((pos[0] - ball_orig[0]) ** 2 +
                                       (pos[1] - ball_orig[1]) ** 2) for pos in ball_poss_rec])

        return loss / batch

    def calculate_ball_loss_slow(self, ball_locations, reconstruction):
        # on entry, ball locations is shape [batch, 2]  example, [[original_image1_ballx,original_image1_bally],[original_image2_ballx,original_image2_bally],...]
        # reconstraction shape [batch, screen_height, screen_width,1]???

        batch_size = ball_locations.shape[0]
        screen_width = reconstruction.shape[2]
        screen_height = reconstruction.shape[1]
        score = 0
        for b in range(batch_size):
            ball_x = ball_locations[b, 0]
            ball_y = ball_locations[b, 1]
            reconstucted_image = reconstruction[b, :, :, 0]
            reconstucted_image_brightness = tf.reduce_sum(
                reconstucted_image*reconstucted_image)+1e-6
            normalised_brightness_image = reconstucted_image/reconstucted_image_brightness
            for x in range(screen_width):
                for y in range(screen_height):
                    score += normalised_brightness_image[x,
                                                         y]*tf.sqrt((x-ball_x)**2+(y-ball_y)**2)
        return score

    @tf.function
    def calculate_ball_loss_fast(self, ball_locations, reconstruction):
        # on entry, ball locations is shape [batch, 2]  example, [[original_image1_ballx,original_image1_bally],[original_image2_ballx,original_image2_bally],...]
        # reconstraction shape [batch, screen_height, screen_width,1]???

        batch_size = ball_locations.shape[0]
        screen_width = reconstruction.shape[2]
        screen_height = reconstruction.shape[1]

        score = tf.zeros([batch_size], tf.float32)
        ball_x = tf.cast(ball_locations[:, 0], tf.float32)
        ball_y = tf.cast(ball_locations[:, 1], tf.float32)
        reconstucted_image = reconstruction[:, :, :, 0]

        # should leave us with a vector of shape [batch_size]
        reconstucted_image_brightness = tf.reduce_sum(
            reconstucted_image, axis=[1, 2])+1e-6
        normalised_brightness_image = reconstucted_image / \
            tf.reshape(reconstucted_image_brightness, [batch_size, 1, 1])

        for x in range(5, 34):
            for y in range(screen_height):
                score += normalised_brightness_image[:, y,
                                                     x]*tf.sqrt((x-ball_x)**2+(y-ball_y)**2)
        return tf.reduce_mean(score)

    # @tf.function

    def calculate_ball_loss_fast2(self, ball_locations, reconstruction, coordinate_array):
        # on entry, ball locations is shape [batch, 2]  example, [[original_image1_ballx,original_image1_bally],[original_image2_ballx,original_image2_bally],...]
        # reconstraction shape [batch, screen_height, screen_width,1]???
        # coordinate_array is shape [batch_size, screen_height, screen_width, 2]

        coordinate_array_temp = coordinate_array

        batch_size = ball_locations.shape[0]
        screen_width = reconstruction.shape[2]
        screen_height = reconstruction.shape[1]
        try:
            assert coordinate_array.shape[0] == batch_size
        except AssertionError:
            # print(
            #    f"Batch Size {batch_size}, Coord array batch {coordinate_array.shape[0]}")
            coordinate_array_temp = coordinate_array[:batch_size]

        assert coordinate_array_temp.shape[1] == screen_height
        assert coordinate_array_temp.shape[2] == screen_width

        ball_locations_r = tf.reshape(ball_locations, [batch_size, 1, 1, 2])
        relative_locations = coordinate_array_temp - \
            tf.cast(ball_locations_r, tf.float32)
        relative_locations_squared = tf.square(relative_locations)
        # will work out x*x+y*y.  Output shape should be [batch_size, screen_height, screen_widht)
        temp = tf.reduce_sum(relative_locations_squared, axis=3)
        # Output shape should be [batch_size, screen_height, screen_widht)
        pythagorean_distances = tf.sqrt(temp)

        # shape should be [batch_size, screen_height, screen_width)
        reconstructed_image = reconstruction[:, :, :, 0]
        # should leave us with a vector of shape [batch_size]

        '''
        reconstucted_image_brightness = tf.reduce_sum(
            reconstucted_image, axis=[1, 2])+1e-6
        normalised_brightness_image = reconstucted_image / \
            tf.reshape(reconstucted_image_brightness, [batch_size, 1, 1])
        '''

        # slice off paddles:
        #reconstructed_image = reconstructed_image[:, :, 5:34]
        #pythagorean_distances = pythagorean_distances[:, :, 5:34]

        # # this will slightly brighten every pixel.  It will ensure
        # a) we can't get a division by zero when normalising the brigtness.
        # b) That the neural network cannot learn to cheat by setting the brigthness of every pixel to zero.
        brightened_reconstructed_image = reconstructed_image+1e-6
        reconstucted_image_brightness = tf.reduce_sum(
            brightened_reconstructed_image, axis=[1, 2])
        normalised_brightness_image = brightened_reconstructed_image / \
            tf.reshape(reconstucted_image_brightness, [batch_size, 1, 1])

        # shape should be [batch_size, screen_height, screen_width)
        score = normalised_brightness_image*pythagorean_distances
        # shape should be [batch_size)
        score = tf.reduce_sum(score, axis=[1, 2])
        result = tf.reduce_mean(score)

        # old_result = self.calculate_ball_loss_fast(
        #    ball_locations, reconstruction)
        #print("Checking: ", result, old_result)
        #assert abs(result-old_result) < 1e-5
        return result

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        data_np = data.numpy()

        ball_locations = []

        for d in data_np:
            loc = find_balls(d, 5, 34)
            if len(loc) < 1:
                loc = [0, 0]
            else:
                loc = loc[0]
            ball_locations.append(loc)

        ball_locations = np.array(ball_locations)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Calculate reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )

            ball_loss = self.calculate_ball_loss_fast2(
                ball_locations, reconstruction, self.coord_array) * 1

            # Calculate KL loss and total loss
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss + ball_loss

        # Get and propagate gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.ball_loss_tracker.update_state(ball_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "ball_loss": self.ball_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        data_np = data.numpy()

        ball_locations = []

        for d in data_np:
            loc = find_balls(d, 5, 34)
            if len(loc) < 1:
                loc = [0, 0]
            else:
                loc = loc[0]
            ball_locations.append(loc)

        ball_locations = np.array(ball_locations)

        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        # Calculate reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            )
        )
        ball_loss = self.calculate_ball_loss_fast2(
            ball_locations, reconstruction, self.coord_array) * 1

        # Calculate KL loss and total loss
        kl_loss = -0.5 * (1 + z_log_var -
                          tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss + ball_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "ball_loss": ball_loss,
        }


class VAEBallTrackNoPaddleInternal(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAEBallTrackNoPaddleInternal, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.diagonal = math.sqrt(
            encoder.input_shape[1] ** 2 + encoder.input_shape[2] ** 2)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.ball_loss_tracker = keras.metrics.Mean(name="ball_loss")
        self.encoder_shape = encoder.input_shape

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def setup_coordinate_array(self, batch):
        self.coord_array = self.build_coordinate_array(
            batch, self.encoder_shape[2], self.encoder_shape[1])

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.kl_loss_tracker,
            self.ball_loss_tracker
        ]

    def build_coordinate_array(self, batch_size, screen_width, screen_height):
        result = np.zeros((screen_height, screen_width, 2), np.float32)
        for x in range(screen_width):  # todo: look up numpy mesh to do this faster?
            for y in range(screen_height):
                result[y, x, 0] = x
                result[y, x, 1] = y
        # should be shape [batch_size, screen_height, screen_width, 2]
        result = np.stack([result]*batch_size, axis=0)
        return tf.constant(result)

    # @tf.function

    def calculate_ball_loss_fast2(self, ball_locations, reconstruction, coordinate_array):
        # on entry, ball locations is shape [batch, 2]  example, [[original_image1_ballx,original_image1_bally],[original_image2_ballx,original_image2_bally],...]
        # reconstraction shape [batch, screen_height, screen_width,1]???
        # coordinate_array is shape [batch_size, screen_height, screen_width, 2]

        coordinate_array_temp = coordinate_array

        batch_size = ball_locations.shape[0]
        screen_width = reconstruction.shape[2]
        screen_height = reconstruction.shape[1]
        try:
            assert coordinate_array.shape[0] == batch_size
        except AssertionError:
            # print(
            #    f"Batch Size {batch_size}, Coord array batch {coordinate_array.shape[0]}")
            coordinate_array_temp = coordinate_array[:batch_size]

        assert coordinate_array_temp.shape[1] == screen_height
        assert coordinate_array_temp.shape[2] == screen_width

        ball_locations_r = tf.reshape(ball_locations, [batch_size, 1, 1, 2])
        relative_locations = coordinate_array_temp - \
            tf.cast(ball_locations_r, tf.float32)
        relative_locations_squared = tf.square(relative_locations)
        # will work out x*x+y*y.  Output shape should be [batch_size, screen_height, screen_widht)
        temp = tf.reduce_sum(relative_locations_squared, axis=3)
        # Output shape should be [batch_size, screen_height, screen_widht)
        pythagorean_distances = tf.sqrt(temp)

        # shape should be [batch_size, screen_height, screen_width)
        reconstructed_image = reconstruction[:, :, :, 0]
        # should leave us with a vector of shape [batch_size]

        '''
        reconstucted_image_brightness = tf.reduce_sum(
            reconstucted_image, axis=[1, 2])+1e-6
        normalised_brightness_image = reconstucted_image / \
            tf.reshape(reconstucted_image_brightness, [batch_size, 1, 1])
        '''

        # slice off paddles:
        #reconstructed_image = reconstructed_image[:, :, 5:34]
        #pythagorean_distances = pythagorean_distances[:, :, 5:34]

        # # this will slightly brighten every pixel.  It will ensure
        # a) we can't get a division by zero when normalising the brigtness.
        # b) That the neural network cannot learn to cheat by setting the brigthness of every pixel to zero.
        brightened_reconstructed_image = reconstructed_image+1e-6
        reconstucted_image_brightness = tf.reduce_sum(
            brightened_reconstructed_image, axis=[1, 2])
        normalised_brightness_image = brightened_reconstructed_image / \
            tf.reshape(reconstucted_image_brightness, [batch_size, 1, 1])

        # shape should be [batch_size, screen_height, screen_width)
        score = normalised_brightness_image*pythagorean_distances
        # shape should be [batch_size)
        score = tf.reduce_sum(score, axis=[1, 2])
        result = tf.reduce_mean(score)

        # old_result = self.calculate_ball_loss_fast(
        #    ball_locations, reconstruction)
        #print("Checking: ", result, old_result)
        #assert abs(result-old_result) < 1e-5
        return result

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        data_np = data.numpy()

        ball_locations = find_balls_tf(data)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            ball_loss = self.calculate_ball_loss_fast2(
                ball_locations, reconstruction, self.coord_array) * 1

            # Calculate KL loss and total loss
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = kl_loss + ball_loss

        # Get and propagate gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update loss
        self.total_loss_tracker.update_state(total_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.ball_loss_tracker.update_state(ball_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "ball_loss": self.ball_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        data_np = data.numpy()

        ball_locations = find_balls_tf(data)

        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        ball_loss = self.calculate_ball_loss_fast2(
            ball_locations, reconstruction, self.coord_array) * 1

        # Calculate KL loss and total loss
        kl_loss = -0.5 * (1 + z_log_var -
                          tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = kl_loss + ball_loss

        return {
            "loss": total_loss,
            "kl_loss": kl_loss,
            "ball_loss": ball_loss,
        }
