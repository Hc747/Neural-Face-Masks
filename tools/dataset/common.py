from keras_preprocessing.image import ImageDataGenerator
from typing import Optional
from constants import RANDOM_STATE


def flow(provider, directory, shape, subset=None, random_state=RANDOM_STATE, class_mode='categorical'):
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
