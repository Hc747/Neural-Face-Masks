import argparse
import os
import cv2
import dlib
import numpy as np
from PIL import Image
from model.face.detectors import FaceDetectorProvider, FaceDetector
from ui.callback.callback import FrameCallback
from ui.processing.image import resize_image
from ui.gui import GUI

# TODO: logging

DEFAULT_IMAGE_SOURCE: str = 'video'
DEFAULT_DETECTOR: str = 'realtime'
DEFAULT_COMPLEX_DETECTOR_PATH: str = os.path.abspath(os.path.join('.', 'model', 'mmod_human_face_detector.dat'))
DEFAULT_WIDTH: int = 1024
DEFAULT_HEIGHT: int = 720
DEFAULT_FRAME_SIZE: int = 360
DEFAULT_FACE_SIZE: int = 224


def configuration():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=DEFAULT_IMAGE_SOURCE, help='Image source (video)')
    parser.add_argument('--detector', default=DEFAULT_DETECTOR, help='Face detector (realtime or accurate)')
    parser.add_argument('--detector_path', default=DEFAULT_COMPLEX_DETECTOR_PATH, help='Face detector model path (complex)')
    parser.add_argument('--width', default=DEFAULT_WIDTH, help='Camera resolution (width)')
    parser.add_argument('--height', default=DEFAULT_HEIGHT, help='Camera resolution (height)')
    parser.add_argument('--frame_size', default=DEFAULT_FRAME_SIZE, help='Frame callback resolution (width and height)')
    parser.add_argument('--face_size', default=DEFAULT_FACE_SIZE, help='Face image size')
    return parser.parse_args(), parser


def get_face(frame, detection, detector: FaceDetector):
    # TODO: return Image?
    # TODO: upscale / resize for input to mask detector...
    return dlib.sub_image(img=frame, rect=detector.rect(detection))


def process_frame(frame, detector: FaceDetector, frame_size: int, face_size: int):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # TODO: resize and center without having to convert to and from an Image...
    frame = np.asarray(resize_image(Image.fromarray(frame), frame_size))

    detections = detector.detect(frame)
    for detection in detections:
        (x0, y0, width, height) = detector.bounding_box(detection)
        x1, y1 = x0 + width, y0 + height

        lower_face = (x0, y0)
        upper_face = (x1, y1)
        cv2.rectangle(frame, lower_face, upper_face, 255, 1)
        # white: bounding box of face

        dx = (face_size - (x1 - x0)) // 2
        dy = (face_size - (y1 - y0)) // 2

        lower_crop = (x0 - dx, y0 - dy)
        upper_crop = (x1 + dx, y1 + dy)
        cv2.rectangle(frame, lower_crop, upper_crop, 0, 1)
        # black: bounding box cropped image to be fed into the mask detector

        print(f'frame_size: {frame_size}, face_size: {face_size}, dx: {dx}, dy: {dy}')
        print(f'lower_face: {lower_face}, upper_face: {upper_face}')
        print(f'lower_crop: {lower_crop}, upper_crop: {upper_crop}')
        # TODO: prevent overflow...
    return Image.fromarray(frame)


def get_detector(config) -> FaceDetector:
    if config.detector == 'accurate':
        return FaceDetectorProvider.complex(config.detector_path)
    return FaceDetectorProvider.simple()


def get_callback(config, detector: FaceDetector) -> FrameCallback:
    frame_size = config.frame_size
    face_size = config.face_size
    return FrameCallback(lambda frame: process_frame(frame, detector, frame_size=frame_size, face_size=face_size))


if __name__ == '__main__':
    args, _ = configuration()
    print(f'Application configuration: {args}')

    detector: FaceDetector = get_detector(args)
    callback: FrameCallback = get_callback(args, detector)

    gui = GUI('Face the facts', width=args.width, height=args.height, callback=callback)
    gui.start()
