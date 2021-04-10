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
DEFAULT_SCALE: int = 360


def configuration():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=DEFAULT_IMAGE_SOURCE, help='Image source (video)')
    parser.add_argument('--detector', default=DEFAULT_DETECTOR, help='Face detector (realtime or accurate)')
    parser.add_argument('--detector_path', default=DEFAULT_COMPLEX_DETECTOR_PATH, help='Face detector model path (complex)')
    parser.add_argument('--width', default=DEFAULT_WIDTH, help='Camera resolution (width)')
    parser.add_argument('--height', default=DEFAULT_HEIGHT, help='Camera resolution (height)')
    parser.add_argument('--scale', default=DEFAULT_SCALE, help='Frame callback resolution (width and height)')
    return parser.parse_args(), parser


def get_face(frame, detection, detector: FaceDetector):
    # TODO: return Image?
    # TODO: upscale / resize for input to mask detector...
    return dlib.sub_image(img=frame, rect=detector.rect(detection))


def process_frame(frame, detector: FaceDetector, size: int):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # TODO: resize and center without having to convert to and from an Image...
    frame = np.asarray(resize_image(Image.fromarray(frame), size))

    detections = detector.detect(frame)
    for detection in detections:
        (x0, y0, width, height) = detector.bounding_box(detection)
        lower = (x0, y0)
        upper = (x0 + width, y0 + height)
        cv2.rectangle(frame, lower, upper, 255, 2)

    return Image.fromarray(frame)


def get_detector(config) -> FaceDetector:
    if config.detector == 'accurate':
        return FaceDetectorProvider.complex(config.detector_path)
    return FaceDetectorProvider.simple()


def get_callback(config, detector: FaceDetector) -> FrameCallback:
    scale = config.scale
    return FrameCallback(lambda frame: process_frame(frame, detector, scale))


if __name__ == '__main__':
    args, _ = configuration()
    print(f'Application configuration: {args}')

    detector: FaceDetector = get_detector(args)
    callback: FrameCallback = get_callback(args, detector)

    gui = GUI('Face the facts', width=args.width, height=args.height, callback=callback)
    gui.start()
