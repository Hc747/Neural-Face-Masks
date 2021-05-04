import math
import cv2
import dlib
import numpy as np
from PIL import Image
from typing import Optional
from keras.models import Model  # TODO: return layer of abstraction
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
    return int(-v // 2 if v < 0 else v // 2 if v > 0 else 0)


def delta_ceil(v: int) -> int:
    return int(math.ceil(-v / 2 if v < 0 else v / 2 if v > 0 else 0))


# TODO: i suck at maths, so this is necessary...
def shift(left: int, top: int, right: int, bottom: int, target: int, frame_width: int, frame_height: int):
    l, t, r, b = 0, 0, 0, 0
    swap: bool = True

    def w() -> int:
        return (right + r) - (left + l)

    def h() -> int:
        return (bottom + b) - (top + t)

    width: int = w()
    while width != target:
        diff: int = target - width
        pos = diff > 0
        swap = not swap

        if swap:
            l = max(0, l - 1 if pos else l + 1)
        else:
            r = min(frame_width, r + 1 if pos else r - 1)
        width = w()

    height: int = h()
    while height != target:
        diff: int = target - height
        pos = diff > 0
        swap = not swap

        if swap:
            t = max(0, t - 1 if pos else t + 1)
        else:
            b = min(frame_height, b + 1 if pos else b - 1)
        height = h()

    le, ri = bind(left + l, right + r, 0, frame_width)
    to, bo = bind(top + t, bottom + b, 0, frame_height)
    return le, to, ri, bo

# dlib cnn detector and then batch classify using
# weak cnn ensemble
# retrain dlib?
# TODO: resize and scale down if face is larger than boundary!!!
def process_frame(frame, face: FaceDetector, mask: Model, match_size: int, resize_to: Optional[int] = None):
    if resize_to is not None:
        frame = crop_square(frame, resize_to)

    # TODO: scale down image for face detection (unless it adversely affects accuracy)
    # TODO: scale up image for mask detection

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.asarray(frame)

    (frame_height, frame_width) = np.shape(frame)[:2]

    images, face_coordinates, crop_coordinates = [], [], []
    masked, unmasked, unknown = 0, 0, 0

    detections = face.detect(frame)  # TODO: train for faces with and without masks
    for detection in detections:
        (face_left, face_top, face_width, face_height) = face.bounding_box(detection)
        face_right, face_bottom = face_left + face_width, face_top + face_height

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

        if is_assertions_enabled():
            expect(lambda: 0 < match_size < frame_width, lambda: f'(0 < face < frame) = 0 < {match_size} < {frame_width}')
            expect(lambda: 0 < match_size < frame_height, lambda: f'(0 < face < frame) = 0 < {match_size} < {frame_height}')
            expect(lambda: 0 <= face_left <= face_right <= frame_width, lambda: f'(0 <= left <= right <= frame) = 0 <= {face_left} <= {face_right} <= {frame_width}')
            expect(lambda: 0 <= face_top <= face_bottom <= frame_height, lambda: f'(0 <= top <= bottom <= frame) = 0 <= {face_top} <= {face_bottom} <= {frame_height}')
            expect(lambda: (crop_right - crop_left) == match_size, lambda: f'(right - left == mask) = ({crop_right} - {crop_left} == {match_size}) = {crop_right - crop_left} == {match_size}')
            expect(lambda: (crop_bottom - crop_top) == match_size, lambda: f'(bottom - top == mask) = ({crop_bottom} - {crop_top} == {match_size}) = {crop_bottom - crop_top} == {match_size}')

        face_inside_crop = face_left >= crop_left and face_top >= crop_top and face_right <= crop_right and face_bottom <= crop_bottom
        face_location = (face_left, face_top, face_right, face_bottom)
        crop_location = shift(crop_left, crop_top, crop_right, crop_bottom, match_size - 1, frame_width - 1, frame_height - 1)

        if face_inside_crop:
            image = dlib.sub_image(img=frame, rect=dlib.rectangle(*crop_location))
        else:
            image = cv2.resize(dlib.sub_image(img=frame, rect=dlib.rectangle(*face_location)), (match_size, match_size))

        (width, height) = np.shape(image)[:2]

        # TODO: more expensive than correctly determining the coordinates...
        # TODO: adequate workaround for now...
        # if width != match_size or height != match_size:
        #     debug(lambda: f'Slooooooooow')
        #     image = cv2.resize(image, (match_size, match_size), interpolation=cv2.INTER_AREA)
        #     (width, height) = np.shape(image)[:2]
        if width != match_size or height != match_size:
            debug(lambda: 'Size mismatch...')

        if width == height == match_size:
            face_coordinates.append(face_location)
            crop_coordinates.append(crop_location)
            images.append(image)
        else:
            unknown += 1
            draw_boxes(frame, face_location, COLOUR_WHITE, 'Unknown', crop_location, COLOUR_BLUE, f'Unprocessable ({width}x{height})')

        if is_debug():
            debug(lambda: '---start---')
            debug(lambda: f'face_x_offset: {face_x_offset}, face_y_offset: {face_y_offset}')
            debug(lambda: f'crop_x_offset: {crop_x_offset}, crop_y_offset: {crop_y_offset}')
            debug(lambda: f'crop_x_ceil: {crop_x_ceil}, crop_y_ceil: {crop_y_ceil}')
            debug(lambda: f'frame_size: {(frame_width, frame_height)}, mask_input_size: {match_size}')
            debug(lambda: f'face_boundary: {face_location}')
            debug(lambda: f'crop_boundary: {crop_location}')
            debug(lambda: f'shape: {width}x{height}')
            debug(lambda: f'face_inside_crop: {face_inside_crop}')
            debug(lambda: '---face---')
            debug(lambda: f'[L {pad(face_left)}, R {pad(face_right)}, W {pad(face_width)}]')
            debug(lambda: f'[T {pad(face_top)}, B {pad(face_bottom)}, H {pad(face_height)}]')
            debug(lambda: '---mask---')
            debug(lambda: f'[L {pad(crop_left)}, R {pad(crop_right)}, W {pad(crop_right - crop_left)}]')
            debug(lambda: f'[T {pad(crop_top)}, B {pad(crop_bottom)}, H {pad(crop_bottom - crop_top)}]')
            debug(lambda: '---end---')
            debug(lambda: '')

    if len(images) > 0:
        images = np.asarray(images)
        predictions = mask.predict(images)

        if is_debug():
            debug(lambda: f'Predictions: {predictions}')

        for index, (p, f, b) in enumerate(zip(predictions, face_coordinates, crop_coordinates)):
            m, u = draw_hit(frame, index, p, f, b)
            masked += m
            unmasked += u

    draw_stats(frame, masked, unmasked, unknown)

    if is_debug():
        debug(lambda: f'Masked faces: {masked}, Unmasked faces: {unmasked}, Unknown faces: {unknown}')

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


def get_callback(config, face: FaceDetector, mask: Model) -> FrameCallback:
    frame_size = config.frame_size
    # frame_size = None

    def fn(frame):
        return process_frame(frame, face, mask, match_size=IMAGE_SIZE, resize_to=frame_size)

    return FrameCallback(fn)


if __name__ == '__main__':
    debug(lambda: f'Application configuration: {args}')
    debug(FaceDetectorProvider.version)
    debug(MaskDetectorProvider.version)

    faces: FaceDetector = FaceDetectorProvider.get_face_detector(args)
    masks: Model = MaskDetectorProvider.get_mask_detector(args)
    callback: FrameCallback = get_callback(args, faces, masks)

    gui = GUI(title=args.title, width=args.width, height=args.height, callback=callback)
    gui.start()
