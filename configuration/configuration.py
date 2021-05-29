import sys
from tensorflow.python.keras import Model
from constants import FACE_DETECTOR_SVM, FACE_DETECTOR_CNN, ALL_FACE_DETECTORS
from detectors.face.detectors import FaceDetector


def debug(message, out=sys.stdout):
    print(message(), file=out)


def expect(condition, message):
    assert condition(), message()


class ApplicationConfiguration:
    __debug: bool
    __assert: bool
    __experiment: bool
    __development: bool

    __svm: FaceDetector
    __cnn: FaceDetector
    __face: FaceDetector
    __mask: Model

    __scale: int

    def __init__(self, config, faces, mask):
        self.__debug = config.debug
        self.__assert = config.enable_assertions
        self.__experiment = config.experimental
        self.__scale = config.scale
        self.__production = config.production
        self.__faces = faces
        self.__mask = mask
        self.face = config.face_detector

    @property
    def debugging(self) -> bool:
        return self.__debug

    @debugging.setter
    def debugging(self, value: bool):
        self.__debug = value

    def toggle_debugging(self):
        self.debugging = not self.debugging

    @property
    def asserting(self) -> bool:
        return self.__assert

    @asserting.setter
    def asserting(self, value: bool):
        self.__assert = value

    def toggle_asserting(self):
        self.asserting = not self.asserting

    @property
    def experimenting(self) -> bool:
        return self.__experiment

    @experimenting.setter
    def experimenting(self, value: bool):
        self.__experiment = value

    def toggle_experimenting(self):
        self.experimenting = not self.experimenting

    @property
    def production(self) -> bool:
        return self.__production

    @production.setter
    def production(self, value: bool):
        self.__production = value

    def toggle_production(self):
        self.production = not self.production

    @property
    def scale(self) -> int:
        return self.__scale

    @scale.setter
    def scale(self, value: int):
        self.__scale = value

    @property
    def svm(self) -> FaceDetector:
        return self.__svm

    @property
    def cnn(self) -> FaceDetector:
        return self.__cnn

    @property
    def face(self) -> FaceDetector:
        return self.__face

    @face.setter
    def face(self, value: str):
        face = self.__faces.get(value, None)
        if face is None:
            raise ValueError(f'Unknown face detector implementation: {value}. Must be one of: {ALL_FACE_DETECTORS}')
        else:
            self.__face = face

    def face_str(self) -> str:
        return self.__face.name()

    @property
    def mask(self) -> Model:
        return self.__mask
