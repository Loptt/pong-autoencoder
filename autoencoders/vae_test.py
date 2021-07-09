from autoencoders.VAE import VAE, VAEBallTrack
from keras.preprocessing.image import ImageDataGenerator

DATASET_SIZE = 11862
INPUT_SHAPE = (40, 40, 1)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True
)
# Allow horizontal flip as a mirror image of a game is a valid game state

train_datagen = datagen.flow_from_directory('images_trans/',
                                            target_size=(
                                                INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                            color_mode='grayscale',
                                            class_mode='input',
                                            shuffle=True,
                                            subset='training',
                                            batch_size=32)

val_datagen = datagen.flow_from_directory('images_trans/',
                                          target_size=(
                                              INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                          color_mode='grayscale',
                                          class_mode='input',
                                          shuffle=True,
                                          subset='validation',
                                          batch_size=32)

vae = VAEBallTrack(
    layers=5,
    input_shape=INPUT_SHAPE,
    latent_size=16,
    kernel_size=5,
    name="VAEP")

vae.summary()
history = vae.train(train_datagen, val_datagen, epochs=30, batch_size=32)
