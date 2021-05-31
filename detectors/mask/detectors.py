import abc
import os.path
import numpy as np
import tensorflow as tf
from typing import Tuple
from tensorflow import keras
from tensorflow.keras import Model
from constants import MASK_DETECTOR_ASHISH, ALL_MASK_DETECTORS, MASK_DETECTOR_CABANI, \
    COLOUR_WHITE, COLOUR_GREEN, COLOUR_RED, COLOUR_YELLOW
from network.network_architecture import NetworkArchitecture, LOSS_WEIGHTS, LOSS_FUNCTIONS


__root: str = os.path.abspath('.')
base: str = os.path.join(__root, 'models', 'mask')


class ResultMapping:
    def __init__(self, idx: int, label: str, confidence: float, colour: Tuple):
        self.__idx = idx
        self.__label = label
        self.__confidence = confidence
        self.__colour = colour

    @property
    def idx(self) -> int:
        return self.__idx

    @property
    def label(self) -> str:
        return self.__label

    @property
    def confidence(self) -> float:
        return self.__confidence

    @property
    def colour(self) -> Tuple:
        return self.__colour

    @staticmethod
    def unknown(idx: int, confidence: float):
        return ResultMapping(idx=idx, label='Unknown', confidence=confidence, colour=COLOUR_WHITE)


class MaskDetector(metaclass=abc.ABCMeta):
    def __init__(self, model: Model, mapping, name: str):
        self.__model = model
        self.__mapping = mapping
        self.__size = len(mapping) + 1  # account for unknown case
        self.__name = name

    @property
    def model(self) -> Model:
        return self.__model

    @property
    def mapping(self):
        return self.__mapping

    @property
    def output_size(self) -> int:
        return self.__size

    def name(self) -> str:
        return self.__name

    def evaluate(self, predictions: [float]) -> ResultMapping:
        values = np.asarray(predictions)
        if values.shape[-1] <= 1:
            raise ValueError('Values in unexpected format: only one indice.')
        index = values.argmax(axis=-1)
        confidence = values[index] * 100.0  # convert to percentage
        mapping = self.__mapping.get(index, None)
        if mapping is None:
            return ResultMapping.unknown(index, confidence)
        return ResultMapping(idx=index, label=mapping.get('label'), confidence=confidence, colour=mapping.get('colour'))


ASHISH_MAPPING = {
    0: {'label': 'Masked', 'colour': COLOUR_GREEN},
    1: {'label': 'Unmasked', 'colour': COLOUR_RED}
}

CABANI_MAPPING = {
    3: {'label': 'Masked', 'colour': COLOUR_GREEN},
    4: {'label': 'Unmasked', 'colour': COLOUR_RED},
    0: {'label': 'Masked - Incorrect (uncovered nose and mouth)', 'colour': COLOUR_YELLOW},
    1: {'label': 'Masked - Incorrect (uncovered nose)', 'colour': COLOUR_YELLOW},
    2: {'label': 'Masked - Incorrect (uncovered chin)', 'colour': COLOUR_YELLOW}
}


class MaskDetectorProvider:
    @staticmethod
    def version() -> str:
        return f'Mask detector: TensorFlow - {tf.__version__}, Keras: {keras.__version__}, GPU(s): {tf.config.list_physical_devices("GPU")}'

    @staticmethod
    def ashish() -> MaskDetector:
        directory = os.path.join(base, 'ashish', 'classification', 'checkpoint')
        model: Model = MaskDetectorProvider.__model(directory)
        return MaskDetector(model=model, mapping=ASHISH_MAPPING, name=MASK_DETECTOR_ASHISH)

    @staticmethod
    def cabani() -> MaskDetector:
        directory = os.path.join(base, 'cabani', 'classification', 'checkpoint')
        model: Model = MaskDetectorProvider.__model(directory)
        return MaskDetector(model=model, mapping=CABANI_MAPPING, name=MASK_DETECTOR_CABANI)

    @staticmethod
    def __model(directory: str) -> Model:
        model = tf.keras.models.load_model(directory)
        if model is None:
            raise ValueError(f'Pre-trained model not found: {directory}')
        return NetworkArchitecture.compile_static(model, loss=LOSS_FUNCTIONS, weights=LOSS_WEIGHTS)

    @staticmethod
    def get_mask_detector(detector: str, **kwargs) -> MaskDetector:
        factory = {
            MASK_DETECTOR_ASHISH: MaskDetectorProvider.ashish,
            MASK_DETECTOR_CABANI: MaskDetectorProvider.cabani
        }.get(detector, None)

        if factory is None:
            raise ValueError(f'Unknown mask detector implementation: {detector}. Must be one of: {ALL_MASK_DETECTORS}')
        return factory(**kwargs)
