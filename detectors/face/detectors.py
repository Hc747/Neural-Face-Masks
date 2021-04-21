import dlib
from imutils import face_utils
from constants import FACE_DETECTOR_SVM, FACE_DETECTOR_CNN, ALL_FACE_DETECTORS

DEFAULT_UPSCALE: int = 0


class FaceDetector:
    def __init__(self, detector, rect, upscale: int = DEFAULT_UPSCALE):
        self.__upscale = upscale
        self.__detector = detector
        self.__rect = rect

    def detect(self, frame):
        return self.__detector(frame, self.__upscale)

    def rect(self, detection):
        return self.__rect(detection)

    def bounding_box(self, detection):
        return face_utils.rect_to_bb(self.rect(detection))


class FaceDetectorProvider:
    @staticmethod
    def version() -> str:
        return f'Face detector: dlib - {dlib.__version__}'

    @staticmethod
    def svm() -> FaceDetector:
        return FaceDetector(dlib.get_frontal_face_detector(), lambda detection: detection)

    @staticmethod
    def cnn(filename: str) -> FaceDetector:
        return FaceDetector(dlib.cnn_face_detection_model_v1(filename=filename), lambda detection: detection.rect)

    @staticmethod
    def get_face_detector(config) -> FaceDetector:
        if config.face_detector == FACE_DETECTOR_SVM:
            return FaceDetectorProvider.svm()
        elif config.face_detector == FACE_DETECTOR_CNN:
            return FaceDetectorProvider.cnn(config.face_detector_path)
        else:
            raise ValueError(f'Unknown face detector implementation: {config.face_detector}. Must be one of: {ALL_FACE_DETECTORS}')
