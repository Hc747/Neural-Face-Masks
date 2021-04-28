from keras_preprocessing.image import ImageDataGenerator
from typing import Optional


def get_standard_augmentor(split: Optional[float] = None) -> ImageDataGenerator:
    return ImageDataGenerator(
        rescale=1. / 255,
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
