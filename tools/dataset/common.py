from keras_preprocessing.image import ImageDataGenerator
from typing import Optional
from constants import RANDOM_STATE

"""
A module providing common functionality for creating and manipulating datasets using the Keras ImageDataGenerator API.
"""


def flow(provider: ImageDataGenerator, directory: str, shape, subset=None, random_state=RANDOM_STATE, class_mode='categorical'):
    """
    Produces an iterable flow of images from an ImageDataGenerator in a manner consistent across different dataset implementations.

    :param provider:
    The ImageDataGenerator we are using to generate images.
    :param directory:
    The directory of the images on the file system.
    :param shape:
    The shape of the images.
    :param subset:
    The subset of this dataset. One of None, 'training' or 'validation'.
    :param random_state:
    The random state to use for reproducible invocations.
    :param class_mode:
    The classmode of the dataset. One of 'binary' or 'categorical'.

    :return:
    A directory iterator yielding Images.
    """
    (w, h, channels) = shape
    color_mode: str = 'grayscale' if channels == 1 else 'rgb'
    return provider.flow_from_directory(
        directory=directory,
        color_mode=color_mode,
        target_size=(w, h),
        seed=random_state,
        class_mode=class_mode,
        subset=subset
    )


def get_standard_augmentor(split: Optional[float] = None) -> ImageDataGenerator:
    """
    Produces an ImageDataGenerator for augmenting training data in a manner that is consistent across different dataset implementations.

    :param split:
    The validation split (if dividing the dataaset into subsets).

    :return:
    An ImageDataGenerator to use for training a model.
    """
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        brightness_range=[0.2, 1.0],
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=split
    )
