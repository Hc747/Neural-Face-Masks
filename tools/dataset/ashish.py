import os
from typing import Tuple
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from constants import RANDOM_STATE, VALIDATION_SPLIT
from network.network_architecture import LOSS_FUNCTIONS, ClassificationNetwork, CLASSIFICATION_NETWORK_NAME, NETWORKS
from tools.dataset.common import get_standard_augmentor

CLASS_MODE = 'categorical'


def flow(provider, directory, shape, subset=None):
    (w, h, channels) = shape
    color_mode: str = 'grayscale' if channels == 1 else 'rgb'
    return provider.flow_from_directory(
        directory=directory,
        color_mode=color_mode,
        target_size=(w, h),
        seed=RANDOM_STATE,
        class_mode=CLASS_MODE,
        subset=subset
    )


def generate(shape: Tuple[int, int, int], network: str, modify_base: bool, _input: str, _output: str):
    base_directory: str = os.path.join(_input, 'kaggle', 'ashishjangra27', 'face-mask-12k-images-dataset')

    training_directory: str = os.path.join(base_directory, 'training')
    training_provider = get_standard_augmentor(split=VALIDATION_SPLIT)
    validation_provider = ImageDataGenerator(rescale=1. / 255, validation_split=VALIDATION_SPLIT)

    testing_directory: str = os.path.join(base_directory, 'validation')  # testing doesn't contain unmasked faces...
    testing_provider = ImageDataGenerator(rescale=1. / 255)

    training = (flow(training_provider, training_directory, shape, subset='training'), None)
    validation = (flow(validation_provider, training_directory, shape, subset='validation'), None)
    testing = (flow(testing_provider, testing_directory, shape), None)

    output = os.path.join(_output, 'classification', 'checkpoint')
    classes = ['Masked', 'Unmasked']
    base = NETWORKS.get(network, VGG16)

    if modify_base:
        architecture = ClassificationNetwork(base=base, shape=shape, classes=classes)
    else:
        architecture = ClassificationNetwork.standard_architecture(network=base, shape=shape, classes=classes)

    model = architecture.compile(LOSS_FUNCTIONS[CLASSIFICATION_NETWORK_NAME], None)

    return (training, validation, testing), (architecture, model, classes, output)
