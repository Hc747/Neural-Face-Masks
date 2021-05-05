import os.path
import tensorflow as tf
from keras import Model
from tensorflow import keras
from constants import MASK_DETECTOR_ANDREW, MASK_DETECTOR_ASHISH, ALL_MASK_DETECTORS
from network.network_architecture import NetworkArchitecture, LOSS_WEIGHTS, LOSS_FUNCTIONS

__root: str = os.path.abspath('.')
base: str = os.path.join(__root, 'models', 'mask')


# TODO: evaluation functions for translating prediction results...
# TODO: ensemble learning / model..?
class MaskDetectorProvider:
    @staticmethod
    def version() -> str:
        return f'Mask detector: TensorFlow - {tf.__version__}, Keras: {keras.__version__}, GPU(s): {tf.config.list_physical_devices("GPU")}'

    @staticmethod
    def andrew() -> Model:
        directory = os.path.join(base, 'andrew', 'classification', 'checkpoint')
        return MaskDetectorProvider.__load(directory)

    @staticmethod
    def ashish() -> Model:
        directory = os.path.join(base, 'ashish', 'classification', 'checkpoint')
        return MaskDetectorProvider.__load(directory)

    @staticmethod
    def __load(directory: str) -> Model:
        model = tf.keras.models.load_model(directory)
        if model is None:
            raise ValueError(f'Pre-trained model not found: {directory}')
        return NetworkArchitecture.compile_static(model, loss=LOSS_FUNCTIONS, weights=LOSS_WEIGHTS)

    # TODO: specific return type (mask detector..)
    @staticmethod
    def get_mask_detector(detector: str, **kwargs):
        if detector == MASK_DETECTOR_ANDREW:
            return MaskDetectorProvider.andrew()
        elif detector == MASK_DETECTOR_ASHISH:
            return MaskDetectorProvider.ashish()
        else:
            raise ValueError(f'Unknown mask detector implementation: {detector}. Must be one of: {ALL_MASK_DETECTORS}')
