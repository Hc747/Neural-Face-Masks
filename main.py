import os
import cv2
import dlib
import numpy as np
from PIL import Image
from detectors.face.detectors import FaceDetectorProvider, FaceDetector
from detectors.mask.detectors import build, training_generator, testing_generator, MaskDetectorProvider
from ui.callback.callback import FrameCallback
from ui.processing.image import resize_image
from ui.gui import GUI
from constants import *
from config import args, debug

# TODO: logging
# TODO: JIT compilation?
# TODO: add more classes
# TODO: relocate..?
PREDICTION_MASKED: int = 0
PREDICTION_UNMASKED: int = 1
DISABLE_ASSERTS: bool = args.disable_assertions


def evaluate_prediction(probabilities: [float]) -> tuple[int, float]:
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
        score: float = values[0][0] * 100
        if score > 50.0:
            index = PREDICTION_UNMASKED
            confidence = score
        else:
            index = PREDICTION_MASKED
            confidence = 100 - score
    return index, confidence


# TODO: documentation
def bind_lower(value: int, threshold: int) -> tuple[int, int]:
    adjustment = threshold - value if value < threshold else 0
    return value, adjustment


# TODO: documentation
def bind_upper(value: int, threshold: int) -> tuple[int, int]:
    adjustment = -(value - threshold) if value > threshold else 0
    return value, adjustment


# TODO: documentation
def bind(v0: int, v1: int, lower: int, upper: int) -> tuple[int, int]:
    v0, lo = bind_lower(v0, lower)
    v1, hi = bind_upper(v1, upper)
    return v0 + hi, v1 + lo


# TODO: documentation
def pad(value: int, size: int = 3) -> str:
    return f'{value}'.ljust(3)


def process_frame(frame, face: FaceDetector, mask, frame_size: int, mask_input_size: int):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # TODO: resize and center without having to convert to and from an Image...
    frame = np.asarray(resize_image(Image.fromarray(frame), frame_size))

    detections = face.detect(frame)
    for idx, detection in enumerate(detections):
        # TODO: ensure crop dimensions are equal to face_size
        # TODO: ensure coordinates do not fall below 0 or above frame_size, and if so, slide across.
        # TODO: account for if face is too close (too large)...

        (face_left, face_top, face_width, face_height) = face.bounding_box(detection)
        face_right, face_bottom = face_left + face_width, face_top + face_height

        # THRESHOLDS
        # top, left = 0,0
        # bottom, right = frame_size, frame_size
        assert DISABLE_ASSERTS or 0 < mask_input_size < frame_size, \
            f'(0 < face < frame) = 0 < {mask_input_size} < {frame_size}'

        face_left, face_right = bind(face_left, face_right, 0, frame_size)
        assert DISABLE_ASSERTS or 0 <= face_left <= face_right <= frame_size, \
            f'(0 <= left <= right <= frame) = 0 <= {face_left} <= {face_right} <= {frame_size}'

        face_top, face_bottom = bind(face_top, face_bottom, 0, frame_size)
        assert DISABLE_ASSERTS or 0 <= face_top <= face_bottom <= frame_size, \
            f'(0 <= top <= bottom <= frame) = 0 <= {face_top} <= {face_bottom} <= {frame_size}'

        mask_dx = mask_input_size - face_width
        mask_dy = mask_input_size - face_height

        # calculate dx
        if mask_dx < 0:  # face is wider than the mask_input_size
            dx = -mask_dx // 2
        elif mask_dx > 0:  # face is narrower than the mask_input_size
            dx = mask_dx // 2
        else:  # the face is the same width as the mask_input_size
            dx = 0

        # calculate dy
        if mask_dy < 0:  # face is taller than the mask_input_size
            dy = -mask_dy // 2
        elif mask_dy > 0:  # the face is shorter than the mask_input_size
            dy = mask_dy // 2
        else:  # the face is the same height as the mask_input_size
            dy = 0

        crop_left, crop_right = face_left - dx, face_right + dx
        offset_x = (mask_input_size - (crop_right - crop_left))

        crop_top, crop_bottom = face_top - dy, face_bottom + dy
        offset_y = (mask_input_size - (crop_bottom - crop_top))

        crop_left, crop_right = bind(crop_left - offset_x, crop_right, 0, frame_size)
        assert DISABLE_ASSERTS or (crop_right - crop_left) == mask_input_size, \
            f'(right - left == mask) = ({crop_right} - {crop_left} == {mask_input_size}) = {crop_right - crop_left} == {mask_input_size}'

        crop_top, crop_bottom = bind(crop_top - offset_y, crop_bottom, 0, frame_size)
        assert DISABLE_ASSERTS or (crop_bottom - crop_top) == mask_input_size, \
            f'(bottom - top == mask) = ({crop_bottom} - {crop_top} == {mask_input_size}) = {crop_bottom - crop_top} == {mask_input_size}'

        lt_face_coordinates, rb_face_coordinates = (face_left, face_top), (face_right, face_bottom)
        lt_crop_coordinates, rb_crop_coordinates = (crop_left, crop_top), (crop_right - 1, crop_bottom - 1)

        face_image_boundary = dlib.rectangle(*lt_crop_coordinates, *rb_crop_coordinates)
        face_image = dlib.sub_image(img=frame, rect=face_image_boundary)
        (width, height, channels) = np.shape(face_image)

        if width == mask_input_size and height == mask_input_size and channels == IMAGE_CHANNELS:
            crop_colour = COLOUR_BLUE
            prediction, _ = evaluate_prediction(mask(np.reshape([face_image], (1, mask_input_size, mask_input_size, IMAGE_CHANNELS))))

            if prediction == PREDICTION_MASKED:
                face_colour = COLOUR_GREEN
                category = 'Masked'
            elif prediction == PREDICTION_UNMASKED:
                face_colour = COLOUR_RED
                category = 'Unmasked'
            else:
                face_colour = COLOUR_WHITE
                category = 'Undetermined'
        else:
            prediction = None
            category = 'Unknown'
            crop_colour = COLOUR_WHITE
            face_colour = COLOUR_WHITE

        index: int = idx + 1
        face_label: str = f'{index}: {category}'
        boundary_label: str = f'{index}: Boundary'

        # draw face bounding box
        cv2.rectangle(frame, lt_face_coordinates, rb_face_coordinates, face_colour, 1)
        # draw classification label next to face bounding box
        cv2.putText(frame, text=face_label, org=(lt_face_coordinates[0], lt_face_coordinates[1] - TEXT_OFFSET), fontFace=FONT_FACE, fontScale=FONT_SCALE, color=face_colour)
        # draw crop bounding box
        cv2.rectangle(frame, lt_crop_coordinates, rb_crop_coordinates, crop_colour, 1)
        # draw info label next to crop bounding box
        crop_label_offset: int = int(-TEXT_OFFSET * 1.5) if np.allclose(lt_crop_coordinates, lt_face_coordinates, atol=TEXT_OFFSET//2) else TEXT_OFFSET
        cv2.putText(frame, text=boundary_label, org=(lt_crop_coordinates[0], lt_crop_coordinates[1] - crop_label_offset), fontFace=FONT_FACE, fontScale=FONT_SCALE, color=crop_colour)

        debug(f'face_image_boundary: (l,t,r,b){face_image_boundary}, face_image_shape: {np.shape(face_image)}')
        debug(f'frame_size: {frame_size}, mask_input_size: {mask_input_size}, dx: {dx}, dy: {dy}, offset_x: {offset_x}, offset_y: {offset_y}')
        debug('---face---')
        debug(f'[L {pad(lt_face_coordinates[0])}, R {pad(rb_face_coordinates[0])}, W {pad(face_width)}]')
        debug(f'[T {pad(lt_face_coordinates[1])}, B {pad(rb_face_coordinates[1])}, H {pad(face_height)}]')
        debug('---face---')
        debug('---mask---')
        debug(f'[L {pad(lt_crop_coordinates[0])}, R {pad(rb_crop_coordinates[0])}, W {pad(mask_dx)}]')
        debug(f'[T {pad(lt_crop_coordinates[1])}, B {pad(rb_crop_coordinates[1])}, H {pad(mask_dy)}]')
        debug('---mask---')
        debug(f'prediction: {prediction}')

    return Image.fromarray(frame)


def get_face_detector(config) -> FaceDetector:
    if config.face_detector == 'accurate':
        return FaceDetectorProvider.complex(config.face_detector_path)
    return FaceDetectorProvider.simple()


def get_mask_detector(config):
    dataset = os.path.join('.', 'data', 'kaggle', 'ashishjangra27', 'face-mask-12k-images-dataset')
    training = training_generator(os.path.join(dataset, 'training'), IMAGE_SIZE, IMAGE_CHANNELS)
    validation = testing_generator(os.path.join(dataset, 'validation'), IMAGE_SIZE, IMAGE_CHANNELS)
    return build(config.mask_detector_path, IMAGE_SIZE, IMAGE_CHANNELS, training, validation)


def get_callback(config, face: FaceDetector, mask) -> FrameCallback:
    frame_size = config.frame_size
    face_size = IMAGE_SIZE
    return FrameCallback(lambda frame: process_frame(frame, face, mask, frame_size=frame_size, mask_input_size=face_size))


if __name__ == '__main__':
    debug(f'Application configuration: {args}')
    debug(FaceDetectorProvider.version())
    debug(MaskDetectorProvider.version())

    masks = get_mask_detector(args)
    faces: FaceDetector = get_face_detector(args)
    callback: FrameCallback = get_callback(args, faces, masks)

    gui = GUI(title=args.title, width=args.width, height=args.height, callback=callback)
    gui.start()
