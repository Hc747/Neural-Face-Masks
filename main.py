import argparse
import os
import cv2
import dlib
import numpy as np
from PIL import Image
from detectors.face.detectors import FaceDetectorProvider, FaceDetector
from detectors.mask.detectors import build, training_generator, testing_generator
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

DEBUG: bool = False


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


def process_frame(frame, face: FaceDetector, mask, frame_size: int, face_size: int):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # TODO: work on GRAYSCALE..?
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # TODO: resize and center without having to convert to and from an Image...
    frame = np.asarray(resize_image(Image.fromarray(frame), frame_size))

    detections = face.detect(frame)
    for detection in detections:
        (x0, y0, width, height) = face.bounding_box(detection)
        x1, y1 = x0 + width, y0 + height

        lower_face_coordinates = (x0, y0)
        upper_face_coordinates = (x1, y1)

        dx = max((face_size - (x1 - x0)) // 2, 0)
        dy = max((face_size - (y1 - y0)) // 2, 0)

        # TODO: ensure equal to size
        lower_crop_coordinates = (x0 - dx, y0 - dy)
        upper_crop_coordinates = (x1 + dx, y1 + dy)

        face_image_boundary = dlib.rectangle(*lower_crop_coordinates, *upper_crop_coordinates)
        face_image = dlib.sub_image(img=frame, rect=face_image_boundary)

        debug(f'face_image_boundary: {face_image_boundary}, shape: {np.shape(face_image)}, face_image: {face_image}')

        try:
            prediction = int(mask(np.reshape([face_image], (1, face_size, face_size, 3)))[0])
        except:
            prediction = None

        # TODO: constants
        if prediction == 1:  # unmasked
            face_colour = (255, 0, 0)
        elif prediction == 0:  # masked
            face_colour = (0, 255, 0)
        else:  # unable to make a prediction (exception occurred)
            face_colour = (255, 255, 255)

        cv2.rectangle(frame, lower_face_coordinates, upper_face_coordinates, face_colour, 1)
        cv2.rectangle(frame, lower_crop_coordinates, upper_crop_coordinates, 0, 1)
        # black: bounding box cropped image to be fed into the mask detector

        debug(f'frame_size: {frame_size}, face_size: {face_size}, dx: {dx}, dy: {dy}')
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
    training = training_generator(os.path.join(dataset, 'training'), IMAGE_SIZE)
    validation = testing_generator(os.path.join(dataset, 'validation'), IMAGE_SIZE)
    return build(config.mask_detector_path, IMAGE_SIZE, IMAGE_CHANNELS, training, validation)


def get_callback(config, face: FaceDetector, mask) -> FrameCallback:
    frame_size = config.frame_size
    face_size = IMAGE_SIZE
    return FrameCallback(lambda frame: process_frame(frame, face, mask, frame_size=frame_size, face_size=face_size))


if __name__ == '__main__':
    args, _ = configuration()
    debug(f'Application configuration: {args}')

    masks = get_mask_detector(args)
    faces: FaceDetector = get_face_detector(args)
    callback: FrameCallback = get_callback(args, faces, masks)

    gui = GUI('Face the facts', width=args.width, height=args.height, callback=callback)
    gui.start()
