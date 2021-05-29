import mediapipe as mp
import abc
import dlib
from typing import Optional
from mediapipe.framework.formats import location_data_pb2
from mediapipe.python.solutions.drawing_utils import RGB_CHANNELS, _normalized_to_pixel_coordinates

from constants import FACE_DETECTOR_SVM, FACE_DETECTOR_CNN, ALL_FACE_DETECTORS, FACE_DETECTOR_MEDIA_PIPE

DEFAULT_UPSCALE: int = 0
EMPTY = []


class FaceDetector(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def detect(self, frame):
        pass

    @abc.abstractmethod
    def bounding_box(self, frame, detection):
        pass

    def confidence(self, detection) -> Optional[float]:
        return None

    @abc.abstractmethod
    def name(self) -> str:
        pass

    def __enter__(self):
        return self

    def __exit__(self):
        pass


class DLIBFaceDetector(FaceDetector):
    def __init__(self, name, inference, rectangle):
        self.__name = name
        self.__inference = inference
        self.__rectangle = rectangle

    def detect(self, frame):
        return self.__inference(frame)

    def bounding_box(self, frame, detection):
        box = self.__rectangle(detection)
        (left, top) = box.left(), box.top()
        (right, bottom) = box.right(), box.bottom()
        (width, height) = right - left, bottom - top
        return left, top, right, bottom, width, height

    def name(self) -> str:
        return self.__name


class MediaPipeFaceDetector(FaceDetector):
    def __init__(self, detector):
        self.__detector = detector

    def __exit__(self, *args):
        self.__detector.__exit__(*args)

    def detect(self, frame):
        result = self.__detector.process(frame)
        return EMPTY if result.detections is None else result.detections

    def bounding_box(self, frame, detection):
        if not detection.location_data:
            return None
        if frame.shape[2] != RGB_CHANNELS:
            raise ValueError('Input image must contain three channel rgb data.')
        image_rows, image_cols, _ = frame.shape

        location = detection.location_data
        if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
            raise ValueError('LocationData must be relative for this extraction function to work.')

        if not location.HasField('relative_bounding_box'):
            return None

        box = location.relative_bounding_box

        minima = _normalized_to_pixel_coordinates(box.xmin, box.ymin, image_cols, image_rows)
        if minima is None:
            return None

        maxima = _normalized_to_pixel_coordinates(box.xmin + box.width, box.ymin + +box.height, image_cols, image_rows)
        if maxima is None:
            return None

        (left, top), (right, bottom) = minima, maxima
        (width, height) = right - left, bottom - top
        return left, top, right, bottom, width, height

    def confidence(self, detection) -> Optional[float]:
        if not detection.score:
            return None
        return detection.score[0]

    def name(self) -> str:
        return FACE_DETECTOR_MEDIA_PIPE


class FaceDetectorProvider:
    @staticmethod
    def version() -> str:
        return f'Face detector: dlib v{dlib.__version__}'

    @staticmethod
    def media_pipe(confidence: float = 0.5) -> FaceDetector:
        detector = mp.solutions.face_detection.FaceDetection(confidence)
        return MediaPipeFaceDetector(detector)

    @staticmethod
    def svm() -> FaceDetector:
        return DLIBFaceDetector(FACE_DETECTOR_SVM, dlib.get_frontal_face_detector(), lambda result: result)

    @staticmethod
    def cnn(filename: str) -> FaceDetector:
        return DLIBFaceDetector(FACE_DETECTOR_CNN, dlib.cnn_face_detection_model_v1(filename=filename), lambda result: result.rect)

    @staticmethod
    def get_face_detector(detector: str, **kwargs) -> FaceDetector:
        factory = {
            FACE_DETECTOR_MEDIA_PIPE: FaceDetectorProvider.media_pipe,
            FACE_DETECTOR_SVM: FaceDetectorProvider.svm,
            FACE_DETECTOR_CNN: FaceDetectorProvider.cnn
        }.get(detector, None)

        if factory is None:
            raise ValueError(f'Unknown face detector implementation: {detector}. Must be one of: {ALL_FACE_DETECTORS}')
        return factory(**kwargs)
