from CAE import ConvolutionalAutoencoder

DATASET_SIZE = 11862
INPUT_SHAPE = (53, 54, 1)

cae = ConvolutionalAutoencoder(
    layers=6,
    input_shape=(53, 54, 1),
    latent_size=32,
    name="CAE")

cae.summary()
