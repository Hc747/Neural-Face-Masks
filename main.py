import argparse
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

# TODO: logging


DEFAULT_IMAGE_SOURCE: str = 'video'
DEFAULT_DETECTOR: str = 'realtime'
DEFAULT_FACE_DETECTOR_PATH: str = os.path.abspath(os.path.join('.', 'models', 'face', 'mmod_human_face_detector.dat'))
DEFAULT_MASK_DETECTOR_PATH: str = os.path.abspath(os.path.join('.', 'models', 'mask', 'cnn.ckpt'))
DEFAULT_WIDTH: int = 1024
DEFAULT_HEIGHT: int = 720
DEFAULT_FRAME_SIZE: int = 360
DEFAULT_FACE_SIZE: int = 224

IMAGE_SIZE: int = 224
IMAGE_CHANNELS: int = 3

COLOUR_RED: tuple[int, int, int] = (255, 0, 0)
COLOUR_GREEN: tuple[int, int, int] = (0, 255, 0)
COLOUR_BLUE: tuple[int, int, int] = (0, 0, 255)
COLOUR_WHITE: tuple[int, int, int] = (255, 255, 255)
COLOUR_BLACK: tuple[int, int, int] = (0, 0, 0)

DEBUG: bool = True


def debug(msg):
    if DEBUG:
        print(msg)


def configuration():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=DEFAULT_IMAGE_SOURCE, help='Image source (video)')
    parser.add_argument('--face_detector', default=DEFAULT_DETECTOR, help='Face detector (realtime or accurate)')
    parser.add_argument('--face_detector_path', default=DEFAULT_FACE_DETECTOR_PATH, help='Face detector models path')
    parser.add_argument('--mask_detector_path', default=DEFAULT_MASK_DETECTOR_PATH, help='Mask detector models path')
    parser.add_argument('--width', default=DEFAULT_WIDTH, help='Camera resolution (width)')
    parser.add_argument('--height', default=DEFAULT_HEIGHT, help='Camera resolution (height)')
    parser.add_argument('--frame_size', default=DEFAULT_FRAME_SIZE, help='Frame callback resolution (width and height)')
    return parser.parse_args(), parser


def get_face(frame, detection, detector: FaceDetector):
    # TODO: return Image?
    # TODO: upscale / resize for input to mask detector...
    return dlib.sub_image(img=frame, rect=detector.rect(detection))


def get_class(proba):
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > 0.5).astype('int32')


def bind_lower(value: int, threshold: int) -> tuple[int, int]:
    if value < threshold:
        return threshold, threshold - value
    return value, 0


def bind_upper(value: int, threshold: int) -> tuple[int, int]:
    if value > threshold:
        return threshold, -(value - threshold)
    return value, 0


def bind(v0: int, v1: int, lower: int, upper: int) -> tuple[int, int]:
    v0, lo = bind_lower(v0, lower)
    v1, hi = bind_upper(v1, upper)

    r0, r1 = v0 + hi, v1 + lo

    # debug(f'lo: {lo}, hi: {hi}, v0: {v0}, v1: {v1}, r0: {r0}, r1: {r1}')

    return r0, r1


def process_frame(frame, face: FaceDetector, mask, frame_size: int, face_size: int):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # TODO: resize and center without having to convert to and from an Image...
    frame = np.asarray(resize_image(Image.fromarray(frame), frame_size))

    detections = face.detect(frame)
    for detection in detections:
        # TODO: ensure crop dimensions are equal to face_size
        # TODO: ensure coordinates do not fall below 0 or above frame_size, and if so, slide across.

        (face_left, face_top, face_width, face_height) = face.bounding_box(detection)
        face_right, face_bottom = face_left + face_width, face_top + face_height

        face_left, face_right = bind(face_left, face_right, 0, frame_size)
        face_top, face_bottom = bind(face_top, face_bottom, 0, frame_size)

        # TODO: ensure deltas are correct
        offset_x = face_right - face_left
        offset_y = face_bottom - face_top

        # TODO: necessary???
        delta_x = max((face_size - (offset_x if offset_x % 2 == 0 else offset_x + 1)) // 2, 0)
        delta_y = max((face_size - (offset_y if offset_y % 2 == 0 else offset_y + 1)) // 2, 0)

        crop_left, crop_right = bind(face_left - delta_x, face_right + delta_x, 0, frame_size)
        crop_top, crop_bottom = bind(face_top - delta_y, face_bottom + delta_y, 0, frame_size)

        lower_face_coordinates, upper_face_coordinates = (face_left, face_top), (face_right, face_bottom)
        lower_crop_coordinates, upper_crop_coordinates = (crop_left, crop_top), (crop_right, crop_bottom)

        # TODO: ensure dimensions are (face_size, face_size, IMAGE_CHANNELS)
        face_image_boundary = dlib.rectangle(*lower_crop_coordinates, *upper_crop_coordinates)
        face_image = dlib.sub_image(img=frame, rect=face_image_boundary)

        try:
            prediction = int(mask(np.reshape([face_image], (1, face_size, face_size, IMAGE_CHANNELS)))[0])
        except:
            prediction = None

        # TODO: use constants
        if prediction == 1:  # unmasked
            face_colour = COLOUR_RED
        elif prediction == 0:  # masked
            face_colour = COLOUR_GREEN
        else:  # unable to make a prediction (exception occurred)
            face_colour = COLOUR_WHITE

        cv2.rectangle(frame, lower_face_coordinates, upper_face_coordinates, face_colour, 1)
        cv2.rectangle(frame, lower_crop_coordinates, upper_crop_coordinates, COLOUR_BLUE, 1)

        debug(f'face_image_boundary: {face_image_boundary}, face_image_shape: {np.shape(face_image)}')
        debug(f'frame_size: {frame_size}, face_size: {face_size}, delta_x: {delta_x}, delta_y: {delta_y}')
        debug(f'lower_face_coordinates: {lower_face_coordinates}, upper_face_coordinates: {upper_face_coordinates}')
        debug(f'lower_crop_coordinates: {lower_crop_coordinates}, upper_crop_coordinates: {upper_crop_coordinates}')
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
    return FrameCallback(lambda frame: process_frame(frame, face, mask, frame_size=frame_size, face_size=face_size))


if __name__ == '__main__':
    args, _ = configuration()
    debug(f'Application configuration: {args}')
    debug(FaceDetectorProvider.version())
    debug(MaskDetectorProvider.version())

    masks = get_mask_detector(args)
    faces: FaceDetector = get_face_detector(args)
    callback: FrameCallback = get_callback(args, faces, masks)

    gui = GUI('Face the facts', width=args.width, height=args.height, callback=callback)
    gui.start()
