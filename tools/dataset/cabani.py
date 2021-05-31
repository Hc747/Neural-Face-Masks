import os
from typing import Tuple
from keras_preprocessing.image import ImageDataGenerator
from constants import VALIDATION_SPLIT
from tools.dataset.common import get_standard_augmentor, flow


def generate(shape: Tuple[int, int, int], _input: str, _output: str):
    base_directory: str = os.path.join(_input, 'github', 'cabani')

    training_provider = get_standard_augmentor(split=VALIDATION_SPLIT)
    validation_provider = ImageDataGenerator(rescale=1./255, validation_split=VALIDATION_SPLIT)

    training = flow(training_provider, base_directory, shape, subset='training')
    validation = flow(validation_provider, base_directory, shape, subset='validation')

    assert training.num_classes == validation.num_classes, f'Sample class mismatch: {training.num_classes} == {validation.num_classes}'
    classes = training.class_indices

    output = os.path.join(_output, 'classification', 'checkpoint')

    return ((training, None), (validation, None), (None, None)), (classes, output)
