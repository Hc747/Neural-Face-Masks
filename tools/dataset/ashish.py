import os
from typing import Tuple
from keras_preprocessing.image import ImageDataGenerator
from constants import VALIDATION_SPLIT
from tools.dataset.common import get_standard_augmentor, flow

"""
A module exporting the Ashish dataset in a format usable with the train_job module.
Dataset: https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset
"""


def generate(shape: Tuple[int, int, int], _input: str, _output: str):
    base_directory: str = os.path.join(_input, 'kaggle', 'ashishjangra27', 'face-mask-12k-images-dataset')

    training_directory: str = os.path.join(base_directory, 'training')
    testing_directory: str = os.path.join(base_directory, 'validation')  # testing doesn't contain unmasked faces...

    training_provider = get_standard_augmentor(split=VALIDATION_SPLIT)
    validation_provider = ImageDataGenerator(rescale=1./255, validation_split=VALIDATION_SPLIT)
    testing_provider = ImageDataGenerator(rescale=1./255)

    training = flow(training_provider, training_directory, shape, subset='training')
    validation = flow(validation_provider, training_directory, shape, subset='validation')
    testing = flow(testing_provider, testing_directory, shape)

    assert training.num_classes == validation.num_classes == testing.num_classes, f'Sample class mismatch: {training.num_classes} == {validation.num_classes} == {testing.num_classes}'
    classes = training.class_indices

    output = os.path.join(_output, 'classification', 'checkpoint')

    return ((training, None), (validation, None), (testing, None)), (classes, output)
