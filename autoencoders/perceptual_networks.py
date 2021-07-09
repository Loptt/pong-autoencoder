import keras.applications
from keras import Input, Model
from keras import layers
from keras import activations
from keras.utils.vis_utils import plot_model


class PerceptualNetwork():

    def __init__(self, model, layer_amount, input_shape):
        self.layer_amount = layer_amount

        if model not in dir(keras.applications):
            raise Exception(f"Unsupported model {model}")

        # print(input_shape)

        network = getattr(keras.applications, model)(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
        )

        input_img = Input(shape=input_shape)
        x = input_img

        for i in range(layer_amount):
            network.layers[i].trainable = False
            x = network.layers[i](x)

        flat = layers.Flatten()(x)
        output = layers.Activation(activations.sigmoid)(flat)

        self.model = Model(
            input_img, output, name="PerceptualNetwork")

        plot_model(self.model, show_shapes=True)

    def predict(self, x):
        return self.model.predict(x)

    def call(self, x):
        return self.model(x)

    def summary(self):
        self.model.summary()
