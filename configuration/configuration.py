import sys
from typing import List, Dict

from constants import ALL_FACE_DETECTORS, ALL_MASK_DETECTORS
from detectors.face.detectors import FaceDetector
from detectors.mask.detectors import MaskDetector


def debug(message, out=sys.stdout):
    print(message(), file=out)


def expect(condition, message):
    assert condition(), message()


class ApplicationConfiguration:
    __debug: bool
    __assert: bool
    __experiment: bool
    __development: bool

    __faces: Dict[str, FaceDetector]
    __masks: Dict[str, MaskDetector]

    __face: FaceDetector
    __mask: MaskDetector

    __scale: float
    __cache_frames: int
    __padding: int

    def __init__(self, config, faces: Dict[str, FaceDetector], masks: Dict[str, MaskDetector]):
        self.debugging = config.debug
        self.asserting = config.enable_assertions
        self.experimenting = config.experimental
        self.scale = config.scale
        self.padding = config.padding
        self.cache_frames = config.cache_frames
        self.production = config.production
        self.__faces = faces
        self.__masks = masks
        self.face = config.face_detector
        self.mask = config.mask_detector

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
    def scale(self) -> float:
        return self.__scale

    @scale.setter
    def scale(self, value: float):
        self.__scale = max(value, 0.1)

    @property
    def padding(self) -> int:
        return self.__padding

    @padding.setter
    def padding(self, value: int):
        self.__padding = max(value, 0)

    @property
    def cache_frames(self) -> int:
        return self.__cache_frames

    @cache_frames.setter
    def cache_frames(self, value: int):
        self.__cache_frames = max(value, 1)

    @property
    def face(self) -> FaceDetector:
        return self.__face

    @face.setter
    def face(self, value: str):
        face = self.__faces.get(value, None)
        if face is None:
            raise ValueError(f'Unknown face detector implementation: {value}. Must be one of: {ALL_FACE_DETECTORS}')
        self.__face = face

    def face_str(self) -> str:
        return self.face.name()

    @property
    def mask(self) -> MaskDetector:
        return self.__mask

    @mask.setter
    def mask(self, value: str):
        mask = self.__masks.get(value, None)
        if mask is None:
            raise ValueError(f'Unknown mask detector implementation: {value}. Must be one of: {ALL_MASK_DETECTORS}')
        self.__mask = mask

    def mask_str(self) -> str:
        return self.mask.name()
