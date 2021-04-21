import os.path
import tensorflow as tf
from keras import Model
from tensorflow import keras
from network.network_architecture import NetworkArchitecture, LOSS_WEIGHTS, LOSS_FUNCTIONS

root: str = os.path.abspath(os.path.join('..', '..', 'models', 'mask'))


# TODO: evaluation functions for translating prediction results...
class MaskDetectorProvider:
    @staticmethod
    def version() -> str:
        return f'Mask detector: TensorFlow - {tf.__version__}, Keras: {keras.__version__}, GPU(s): {tf.config.list_physical_devices("GPU")}'

    @staticmethod
    def andrew() -> Model:
        directory = os.path.join(root, 'andrew', 'classification', 'checkpoint')
        return MaskDetectorProvider.__load(directory)

    @staticmethod
    def ashish() -> Model:
        directory = os.path.join(root, 'ashish', 'classification', 'checkpoint')
        return MaskDetectorProvider.__load(directory)

    @staticmethod
    def __load(directory: str) -> Model:
        model = tf.keras.models.load_model(directory)
        if model is None:
            raise ValueError(f'Pre-trained model not found: {directory}')
        return NetworkArchitecture.compile_static(model, loss=LOSS_FUNCTIONS, weights=LOSS_WEIGHTS)
