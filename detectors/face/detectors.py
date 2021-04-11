import dlib
from imutils import face_utils


DEFAULT_UPSCALE: int = 1


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
    def simple() -> FaceDetector:
        return FaceDetector(dlib.get_frontal_face_detector(), lambda detection: detection)

    @staticmethod
    def complex(filename: str) -> FaceDetector:
        return FaceDetector(dlib.cnn_face_detection_model_v1(filename=filename), lambda detection: detection.rect)
