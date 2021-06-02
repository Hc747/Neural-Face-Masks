import os.path
import numpy as np
import tensorflow as tf
from typing import Tuple
from tensorflow import keras
from tensorflow.keras import Model
from constants import MASK_DETECTOR_ASHISH, ALL_MASK_DETECTORS, MASK_DETECTOR_CABANI, \
    COLOUR_WHITE, COLOUR_GREEN, COLOUR_RED, COLOUR_YELLOW
from network.network_architecture import NetworkArchitecture, LOSS_WEIGHTS, LOSS_FUNCTIONS

"""
A module exporting face mask detection capabilities through a common MaskDetector interface.
"""


__root: str = os.path.abspath('.')
base: str = os.path.join(__root, 'models', 'mask')


class MaskDetectionResult:
    """
    An object that encapsulates the result of inference by a MaskDetector.
    """

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
        """
        A human-intelligible label describing the result of this inference.
        :return:
        """
        return self.__label

    @property
    def confidence(self) -> float:
        """
        The confidence (0.0-100.0) of this inference.
        """
        return self.__confidence

    @property
    def colour(self) -> Tuple:
        """
        The colour to render this result in.
        """
        return self.__colour

    @staticmethod
    def unknown(idx: int, confidence: float):
        return MaskDetectionResult(idx=idx, label='Unknown', confidence=confidence, colour=COLOUR_WHITE)


class MaskDetector:
    """
    An interface that allows for the classification of images and translation of detection results.
    """

    def __init__(self, model: Model, mapping, name: str):
        self.__model = model
        self.__mapping = mapping
        self.__name = name

    @property
    def model(self) -> Model:
        """
        The keras model used for prediction.
        """
        return self.__model

    @property
    def mapping(self):
        """
        The dictionary mapping prediction indices to labels and colours.
        """
        return self.__mapping

    def name(self) -> str:
        """
        The name of this MaskDetector.
        One of MASK_DETECTOR_ASHISH or MASK_DETECTOR_CABANI.
        """
        return self.__name

    def evaluate(self, predictions: [float]) -> MaskDetectionResult:
        """
        Converts an array of prediction confidences into MaskDetectionResults suitable for display to the end user.
        :param predictions:
        The array of prediction confidences produced by the underlying model
        :return:
        A MaskDetectionResult object denoting the predicted score
        """
        values = np.asarray(predictions)
        if values.shape[-1] <= 1:
            raise ValueError('Values in unexpected format: only one indice.')
        index = values.argmax(axis=-1)
        confidence = values[index] * 100.0  # convert to percentage
        mapping = self.__mapping.get(index, None)
        if mapping is None:
            return MaskDetectionResult.unknown(index, confidence)
        return MaskDetectionResult(idx=index, label=mapping.get('label'), confidence=confidence, colour=mapping.get('colour'))


# key-value pairs denoting prediction indices => (label, colour) for the Ashish and Cabani datasets
ASHISH_MAPPING = {
    0: {'label': 'Masked', 'colour': COLOUR_GREEN},
    1: {'label': 'Unmasked', 'colour': COLOUR_RED}
}

CABANI_MAPPING = {
    3: {'label': 'Masked', 'colour': COLOUR_GREEN},
    4: {'label': 'Unmasked', 'colour': COLOUR_RED},
    0: {'label': 'Uncovered nose and mouth', 'colour': COLOUR_YELLOW},
    1: {'label': 'Uncovered nose', 'colour': COLOUR_YELLOW},
    2: {'label': 'Uncovered chin', 'colour': COLOUR_YELLOW}
}


class MaskDetectorProvider:
    """
    A factory object for instantiating various mask detection models.
    """

    @staticmethod
    def version() -> str:
        return f'Mask detector: TensorFlow - {tf.__version__}, Keras: {keras.__version__}, GPU(s): {tf.config.list_physical_devices("GPU")}'

    @staticmethod
    def ashish() -> MaskDetector:
        """
        Instantiates a MaskDetector trained on the Ashish dataset.
        """
        directory = os.path.join(base, 'ashish', 'classification', 'checkpoint')
        model: Model = MaskDetectorProvider.__load_model(directory)
        return MaskDetector(model=model, mapping=ASHISH_MAPPING, name=MASK_DETECTOR_ASHISH)

    @staticmethod
    def cabani() -> MaskDetector:
        """
        Instantiates a MaskDetector trained on the Cabani dataset.
        """
        directory = os.path.join(base, 'cabani', 'classification', 'checkpoint')
        model: Model = MaskDetectorProvider.__load_model(directory)
        return MaskDetector(model=model, mapping=CABANI_MAPPING, name=MASK_DETECTOR_CABANI)

    @staticmethod
    def __load_model(directory: str) -> Model:
        """
        Loads a pre-trained TensorFlow/Keras model.
        :param directory:
        The directory to load the model (checkpoint) from.
        :return:
        A compiled Keras model instantiated with the checkpointed weights and biases.
        """
        model = tf.keras.models.load_model(directory)
        if model is None:
            raise ValueError(f'Pre-trained model not found: {directory}')
        return NetworkArchitecture.compile_static(model, loss=LOSS_FUNCTIONS, weights=LOSS_WEIGHTS)

    @staticmethod
    def get_mask_detector(detector: str, **kwargs) -> MaskDetector:
        """
        A factory method that takes a string identifier and any number of keyword arguments in order to export a
        MaskDetector object.
        :param detector:
        The type of detector to instantiate.
        :param kwargs:
        The keyword arguments passed into the factory method.
        :raises ValueError:
        If the detector parameter is unrecognised.
        """
        factory = {
            MASK_DETECTOR_ASHISH: MaskDetectorProvider.ashish,
            MASK_DETECTOR_CABANI: MaskDetectorProvider.cabani
        }.get(detector, None)

        if factory is None:
            raise ValueError(f'Unknown mask detector implementation: {detector}. Must be one of: {ALL_MASK_DETECTORS}')
        return factory(**kwargs)
