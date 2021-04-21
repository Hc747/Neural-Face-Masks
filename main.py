import cv2
import dlib
import numpy as np
from PIL import Image
from typing import Optional
from config import args, debug, expect, is_debug, is_assertions_enabled
from constants import *
from detectors.face.detectors import FaceDetectorProvider, FaceDetector
from detectors.mask.detectors import MaskDetectorProvider
from ui.callback.callback import FrameCallback
from ui.gui import GUI
from ui.processing.image import crop_square
from ui.rendering.rendering import draw_stats, draw_boxes

# TODO: logging
# TODO: JIT compilation?
# TODO: add more classes
# TODO: relocate..?

PREDICTION_MASKED: int = 0
PREDICTION_UNMASKED: int = 1


def evaluate_prediction(probabilities: [float]) -> Tuple[int, float]:
    """
    :param probabilities: the prediction scores of each class
    :return: a tuple containing the class indice and the 'confidence'
    """
    values = np.asarray(probabilities)
    if values.shape[-1] > 1:
        index = values.argmax(axis=-1)
        confidence = values[index]
    else:
        # TODO: ensure confidence is correct...
        score: float = values[0] * 100
        if score > 50.0:
            index = PREDICTION_UNMASKED
            confidence = score
        else:
            index = PREDICTION_MASKED
            confidence = 100 - score
    return index, confidence


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
    return -v // 2 if v < 0 else v // 2 if v > 0 else 0


# dlib cnn detector and then batch classify using
# weak cnn ensemble
# retrain dlib?
def process_frame(frame, face: FaceDetector, mask, match_size: int, resize_to: Optional[int] = None):
    if resize_to is not None:
        frame = crop_square(frame, resize_to)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.asarray(frame)

    (frame_height, frame_width) = np.shape(frame)[:2]

    images = []
    coordinates = []
    boundaries = []

    detections = face.detect(frame)  # TODO: train for faces with and without masks
    for detection in detections:
        (face_left, face_top, face_width, face_height) = face.bounding_box(detection)
        face_right, face_bottom = face_left + face_width, face_top + face_height

        face_left, face_right = bind(face_left, face_right, 0, frame_width)
        face_top, face_bottom = bind(face_top, face_bottom, 0, frame_height)

        # TODO: account for if face is too close (too large)...
        dx: int = delta(match_size - face_width)
        dy: int = delta(match_size - face_height)

        crop_left, crop_right = face_left - dx, face_right + dx
        offset_x = (match_size - (crop_right - crop_left))

        crop_top, crop_bottom = face_top - dy, face_bottom + dy
        offset_y = (match_size - (crop_bottom - crop_top))

        crop_left, crop_right = bind(crop_left - offset_x, crop_right, 0, frame_width)
        crop_top, crop_bottom = bind(crop_top - offset_y, crop_bottom, 0, frame_height)

        if is_assertions_enabled():
            expect(lambda: 0 < match_size < frame_width, lambda: f'(0 < face < frame) = 0 < {match_size} < {frame_width}')
            expect(lambda: 0 < match_size < frame_height, lambda: f'(0 < face < frame) = 0 < {match_size} < {frame_height}')
            expect(lambda: 0 <= face_left <= face_right <= frame_width, lambda: f'(0 <= left <= right <= frame) = 0 <= {face_left} <= {face_right} <= {frame_width}')
            expect(lambda: 0 <= face_top <= face_bottom <= frame_height, lambda: f'(0 <= top <= bottom <= frame) = 0 <= {face_top} <= {face_bottom} <= {frame_height}')
            expect(lambda: (crop_right - crop_left) == match_size, lambda: f'(right - left == mask) = ({crop_right} - {crop_left} == {match_size}) = {crop_right - crop_left} == {match_size}')
            expect(lambda: (crop_bottom - crop_top) == match_size, lambda: f'(bottom - top == mask) = ({crop_bottom} - {crop_top} == {match_size}) = {crop_bottom - crop_top} == {match_size}')

        f = (face_left, face_top, face_right, face_bottom)
        b = (crop_left, crop_top, crop_right - 1, crop_bottom - 1)
        i = dlib.sub_image(img=frame, rect=dlib.rectangle(*b))

        (width, height) = np.shape(i)[:2]

        if width == height == match_size:
            coordinates.append(f)
            boundaries.append(b)
            images.append(i)
        else:
            draw_boxes(frame, f, COLOUR_WHITE, '', b, COLOUR_WHITE, f'Unprocessable ({width}x{height})')

        if is_debug():
            debug(lambda: f'face_image_boundary: (l,t,r,b){f}, face_image_shape: {np.shape(i)}')
            debug(lambda: f'frame_size: {(frame_width, frame_height)}, mask_input_size: {match_size}, dx: {dx}, dy: {dy}, offset_x: {offset_x}, offset_y: {offset_y}')
            debug(lambda: '---face---')
            debug(lambda: f'[L {pad(face_left)}, R {pad(face_right)}, W {pad(face_width)}]')
            debug(lambda: f'[T {pad(face_top)}, B {pad(face_bottom)}, H {pad(face_height)}]')
            debug(lambda: '---face---')
            debug(lambda: '---mask---')
            debug(lambda: f'[L {pad(crop_left)}, R {pad(crop_right)}]')
            debug(lambda: f'[T {pad(crop_top)}, B {pad(crop_bottom)}]')
            debug(lambda: '---mask---')

    if len(images) <= 0:
        if is_debug():
            debug(lambda: f'No faces detected - short circuiting')
        return Image.fromarray(frame)

    images = np.asarray(images)
    predictions = mask.predict(images)

    if is_debug():
        debug(lambda: f'Predictions: {predictions}')

    masked, unmasked = 0, 0
    for index, (p, f, b) in enumerate(zip(predictions, coordinates, boundaries)):
        m, u = draw_hit(frame, index, p, f, b)
        masked += m
        unmasked += u

    draw_stats(frame, masked, unmasked)

    if is_debug():
        debug(lambda: f'Masked faces: {masked}, Unmasked faces: {unmasked}')

    return Image.fromarray(frame)


def draw_hit(frame, index, prediction, face, boundary):
    prediction, confidence = evaluate_prediction(prediction)
    masked, unmasked = 0, 0

    if prediction == PREDICTION_MASKED:
        face_colour = COLOUR_GREEN
        category = 'Masked'
        masked = 1
    elif prediction == PREDICTION_UNMASKED:
        face_colour = COLOUR_RED
        category = 'Unmasked'
        unmasked = 1
    else:
        face_colour = COLOUR_WHITE
        category = 'Undetermined'

    face_label = f'{index + 1}: {category} - {confidence * 100.0:.02f}%'
    boundary_label = f'{index + 1}: Boundary'

    draw_boxes(frame, face, face_colour, face_label, boundary, COLOUR_BLUE, boundary_label)
    return masked, unmasked


def get_callback(config, face: FaceDetector, mask) -> FrameCallback:
    frame_size = config.frame_size

    def fn(frame):
        return process_frame(frame, face, mask, match_size=IMAGE_SIZE, resize_to=frame_size)

    return FrameCallback(fn)


if __name__ == '__main__':
    debug(lambda: f'Application configuration: {args}')
    debug(FaceDetectorProvider.version)
    debug(MaskDetectorProvider.version)

    faces: FaceDetector = FaceDetectorProvider.get_face_detector(args)
    masks = MaskDetectorProvider.get_mask_detector(args)
    callback: FrameCallback = get_callback(args, faces, masks)

    gui = GUI(title=args.title, width=args.width, height=args.height, callback=callback)
    gui.start()
