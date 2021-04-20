import os
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from constants import RANDOM_STATE
from network.network_architecture import LOSS_FUNCTIONS, ClassificationNetwork, CLASSIFICATION_NETWORK_NAME, NETWORKS

CLASS_MODE = 'categorical'


def flow(provider, directory, shape):
    (w, h, channels) = shape
    color_mode: str = 'grayscale' if channels == 1 else 'rgb'
    return provider.flow_from_directory(
        directory=directory,
        color_mode=color_mode,
        target_size=(w, h),
        seed=RANDOM_STATE,
        class_mode=CLASS_MODE
    )


def generate(shape, _input, _output, args):
    base_directory: str = os.path.join(_input, 'kaggle', 'ashishjangra27', 'face-mask-12k-images-dataset')

    training_directory: str = os.path.join(base_directory, 'training')
    training_provider = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_directory: str = os.path.join(base_directory, 'validation')
    validation_provider = ImageDataGenerator(rescale=1. / 255)

    x = flow(training_provider, training_directory, shape)
    y = None
    validation = flow(validation_provider, validation_directory, shape)

    output = os.path.join(_output, 'classification', 'checkpoint')
    classes = ['Masked', 'Unmasked']
    base = NETWORKS.get(args.network, VGG16)
    architecture = ClassificationNetwork(base=base, shape=shape, classes=classes)
    model = architecture.compile(LOSS_FUNCTIONS[CLASSIFICATION_NETWORK_NAME], None)

    return (x, y, validation), (architecture, model, classes, output)
