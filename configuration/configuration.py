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

    __svm: FaceDetector
    __cnn: FaceDetector
    __face: FaceDetector
    __mask: Model

    __scale: int

    def __init__(self, config, svm: FaceDetector, cnn: FaceDetector, mask: Model):
        self.__debug = config.debug
        self.__assert = config.enable_assertions
        self.__experiment = config.experimental
        self.__scale = config.scale
        self.__svm = svm
        self.__cnn = cnn
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
        if value == FACE_DETECTOR_SVM:
            self.__face = self.__svm
        elif value == FACE_DETECTOR_CNN:
            self.__face = self.__cnn
        else:
            raise ValueError(f'Unknown face detector implementation: {value}. Must be one of: {ALL_FACE_DETECTORS}')

    def face_str(self) -> str:
        if self.face == self.__svm:
            return FACE_DETECTOR_SVM
        elif self.face == self.__cnn:
            return FACE_DETECTOR_CNN
        else:
            return 'Unknown'

    @property
    def mask(self) -> Model:
        return self.__mask
