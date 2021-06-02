import math
import cv2
import dlib
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
from app.configuration.configuration import ApplicationConfiguration
from constants import COLOUR_BLUE, IMAGE_SIZE, MAX_BATCH_SIZE
from detectors.face.detectors import FaceDetector
from detectors.mask.detectors import MaskDetector, MaskDetectionResult
from app.callback.callback import FrameCallback
from app.processing.image import rescale, translate_scale, adjust_bounding_box
from app.rendering.rendering import draw_boxes, draw_stats, display_confidence, draw_floating_head


# TODO: documentation
def bind_lower(value: int, threshold: int) -> Tuple[int, int]:
    adjustment = threshold - value if value < threshold else 0
    return value, adjustment


# TODO: documentation
def bind_upper(value: int, threshold: int) -> Tuple[int, int]:
    adjustment = -(value - threshold) if value > threshold else 0
    return value, adjustment


# TODO: documentation
def bind(v0: int, v1: int, lower: int, upper: int) -> Tuple[int, int]:
    v0, lo = bind_lower(v0, lower)
    v1, hi = bind_upper(v1, upper)
    return v0 + hi, v1 + lo


# TODO: documentation
def pad(value: int, size: int = 3) -> str:
    return f'{value}'.ljust(size)


def delta(v: int) -> int:
    return int(-v // 2 if v < 0 else v // 2 if v > 0 else 0)


def delta_ceil(v: int) -> int:
    return int(math.ceil(-v / 2 if v < 0 else v / 2 if v > 0 else 0))


def adjust(target: int, value: int, x: int, y: int, lower: int, upper: int) -> Tuple[int, int]:
    d: float = (target - value) / 2
    dx: int = int(math.ceil(d))
    dy: int = int(math.floor(d))
    return max(lower, x + (-1 * dx)), min(upper, y + dy)


def shift(left: int, top: int, right: int, bottom: int, target: int, frame_width: int, frame_height: int):
    l, t, r, b = 0, 0, 0, 0

    def w() -> int:
        return (right + r) - (left + l)

    def h() -> int:
        return (bottom + b) - (top + t)

    width: int = w()
    while width != target:
        l, r = adjust(target, width, l, r, 0, frame_width)
        width = w()

    height: int = h()
    while height != target:
        t, b = adjust(target, height, t, b, 0, frame_height)
        height = h()

    le, ri = bind(left + l, right + r, 0, frame_width)
    to, bo = bind(top + t, bottom + b, 0, frame_height)

    return le, to, ri, bo


class DetectionResult:
    def __init__(self, ok: bool, confidence: Optional[float], face, crop, image, width: int, height: int):
        self.__ok = ok
        self.__confidence = confidence
        self.__face = face
        self.__crop = crop
        self.__image = image
        self.__width = width
        self.__height = height

    @property
    def ok(self):
        return self.__ok

    @property
    def confidence(self):
        return self.__confidence

    @property
    def face(self):
        return self.__face

    @property
    def box(self):
        return self.__crop

    @property
    def image(self):
        return self.__image

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    def draw(self, frame, mask: MaskDetector, index, prediction) -> MaskDetectionResult:
        result: MaskDetectionResult = mask.evaluate(prediction)
        idx: int = index + 1

        face_label = f'{idx}: Face - {result.label} - {display_confidence(result.confidence)}'
        box_label = f'{idx}: Boundary - {display_confidence(self.confidence)}'

        draw_boxes(frame, self.face, result.colour, face_label, self.box, COLOUR_BLUE, box_label)
        return result


class ApplicationCallback(FrameCallback):
    __ticks: int = 0
    __previous = (None, None)

    def __init__(self, configuration: ApplicationConfiguration):
        self.__configuration = configuration

    @property
    def ticks(self):
        return self.__ticks

    @property
    def cache(self):
        return self.__configuration.cache_frames

    @property
    def padding(self):
        return self.__configuration.padding

    @property
    def previous(self):
        return self.__previous

    @property
    def face(self) -> FaceDetector:
        return self.__configuration.face

    @property
    def mask(self) -> MaskDetector:
        return self.__configuration.mask

    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2 produces frames as BGR
        frame, scaled, scale = rescale(frame, scale=self.__configuration.scale)
        frame.flags.writeable = True
        scaled.flags.writeable = False  # pass by reference
        return np.asarray(frame), scaled, scale

    def extract_detection(self, face: FaceDetector, detection, frame, scaled, scale: float, padding: int, match_size: int, frame_width: int, frame_height: int) -> Optional[DetectionResult]:
        box = face.bounding_box(scaled, detection)

        if box is None:
            return None

        box_confidence = face.confidence(detection)

        face_left, face_top, face_right, face_bottom, face_width, face_height = translate_scale(box, scale)

        if padding > 0:
            face_left, face_top, face_right, face_bottom, face_width, face_height = adjust_bounding_box(face_left, face_top, face_right, face_bottom, padding)

        face_left, face_right = bind(face_left, face_right, 0, frame_width)
        face_top, face_bottom = bind(face_top, face_bottom, 0, frame_height)

        face_x_offset: int = delta(match_size - face_width)
        face_y_offset: int = delta(match_size - face_height)

        crop_left, crop_right = face_left - face_x_offset, face_right + face_x_offset
        crop_top, crop_bottom = face_top - face_y_offset, face_bottom + face_y_offset

        vx: int = match_size - (crop_right - crop_left)
        vy: int = match_size - (crop_bottom - crop_top)

        crop_x_offset, crop_x_ceil = delta(vx), delta_ceil(vx)
        crop_y_offset, crop_y_ceil = delta(vy), delta_ceil(vy)

        crop_left, crop_right = bind(crop_left + crop_x_offset, crop_right - crop_x_offset, 0, frame_width)
        crop_top, crop_bottom = bind(crop_top + crop_y_offset, crop_bottom - crop_y_offset, 0, frame_height)

        face_inside_crop = face_left >= crop_left and face_top >= crop_top and face_right <= crop_right and face_bottom <= crop_bottom
        face_location = (face_left, face_top, face_right, face_bottom)
        crop_location = shift(crop_left, crop_top, crop_right, crop_bottom, match_size - 1, frame_width, frame_height)

        if face_inside_crop:
            image = dlib.sub_image(img=frame, rect=dlib.rectangle(*crop_location))
        else:
            image = cv2.resize(dlib.sub_image(img=frame, rect=dlib.rectangle(*face_location)), (match_size, match_size))

        image = np.copy(image)
        # prevent modifications to the original area of the image from affecting the 'copy'

        (width, height) = np.shape(image)[:2]
        ok = match_size == width == height
        return DetectionResult(ok=ok, confidence=box_confidence, face=face_location, crop=crop_location, image=image, width=width, height=height)

    def detect(self, face: FaceDetector, frame, scaled, scale: float, padding: int, match_size: int, frame_width: int, frame_height: int) -> List[DetectionResult]:
        output: List[DetectionResult] = []
        detections = face.detect(scaled)
        for detection in detections:
            extracted: Optional[DetectionResult] = self.extract_detection(face, detection, frame, scaled, scale, padding, match_size, frame_width, frame_height)
            if extracted is None:
                continue
            output.append(extracted)
        return output

    def classify(self, mask: MaskDetector, detections: List[DetectionResult]):
        pending = [(index, detection) for (index, detection) in enumerate(detections) if detection.ok]
        hits, total = len(pending), len(detections)
        output = [None] * total

        if hits <= 0:
            return output

        images = np.array(np.asarray([detection.image for (_, detection) in pending]))
        predictions = mask.model.predict(images, batch_size=min(hits, MAX_BATCH_SIZE))

        for source, (destination, _) in enumerate(pending):
            prediction = predictions[source]
            output[destination] = prediction

        return output

    def render(self, frame, mask: MaskDetector, detections, predictions):
        stats = {}
        for index, (detection, prediction) in enumerate(zip(detections, predictions)):
            result: MaskDetectionResult = detection.draw(frame, mask, index, prediction)
            current: int = stats.get(result.label, 0)
            stats[result.label] = current + 1
            draw_floating_head(frame, detection.image, result.colour, index, items=8, size=64, height_offset=80, width_offset=16)

        draw_stats(frame, mask, stats)
        return frame

    def invoke(self, frame) -> Image:
        # phase 0: setup attributes
        face: FaceDetector = self.face
        mask: MaskDetector = self.mask
        # store face and mask detectors locally in case they're updated externally during processing
        ticks, cache, padding = self.ticks, self.cache, self.padding

        # phase 1: pre-processing
        frame, scaled, scale = self.preprocess(frame)
        (frame_height, frame_width) = np.asarray(np.shape(frame)[:2]) - 1
        # offset by one as array's are zero-indexed

        # phase 2: inference (face detection)
        detections = self.detect(face, frame, scaled, scale, padding, match_size=IMAGE_SIZE, frame_width=frame_width, frame_height=frame_height)

        # phase 3: inference (mask detection)
        detector, predictions = self.previous
        # retrieve cached predictions
        update_predictions: bool = (cache <= 0) or (ticks <= 0) or (detector is None or predictions is None) or (detector != mask) or (len(detections) != len(predictions))
        if update_predictions:
            # update cached predictions if they've gone stale
            predictions = self.classify(mask, detections)
            self.__previous = (mask, predictions)

        # phase 4: render results
        frame = self.render(frame, mask, detections, predictions)

        # phase 5: post-processing
        self.__ticks = (ticks + 1) % cache

        return Image.fromarray(frame)
