import argparse
import os
from distutils.util import strtobool

DEFAULT_TITLE: str = 'FTF: Face Mask Analyser'
DEFAULT_IMAGE_SOURCE: str = 'video'
DEFAULT_DETECTOR: str = 'realtime'
DEFAULT_FACE_DETECTOR_PATH: str = os.path.abspath(os.path.join('.', 'models', 'face', 'mmod_human_face_detector.dat'))
DEFAULT_MASK_DETECTOR_PATH: str = os.path.abspath(os.path.join('.', 'models', 'mask', 'cnn.ckpt'))
DEFAULT_WIDTH: int = 1024
DEFAULT_HEIGHT: int = 720
DEFAULT_FRAME_SIZE: int = 360
DEFAULT_FACE_SIZE: int = 224
DEFAULT_DEBUG: bool = True
DEFAULT_DISABLE_ASSERTIONS: bool = True


def boolean(v: str) -> bool:
    return bool(strtobool(v))


# TODO: type safety
# TODO: mutual exclusion
__parser = argparse.ArgumentParser()
__parser.add_argument('--source', default=DEFAULT_IMAGE_SOURCE, help='Image source (video)')
__parser.add_argument('--face_detector', default=DEFAULT_DETECTOR, help='Face detector (realtime or accurate)')
__parser.add_argument('--face_detector_path', default=DEFAULT_FACE_DETECTOR_PATH, help='Face detector models path')
__parser.add_argument('--mask_detector_path', default=DEFAULT_MASK_DETECTOR_PATH, help='Mask detector models path')
__parser.add_argument('--debug', type=boolean, default=DEFAULT_DEBUG, help='Print debug statements (True/False)')
__parser.add_argument('--disable_assertions', type=boolean, default=DEFAULT_DISABLE_ASSERTIONS, help='Disable assertions at runtime (True/False)')
__parser.add_argument('--title', default=DEFAULT_TITLE, help='Application GUI title')
__parser.add_argument('--width', default=DEFAULT_WIDTH, help='Camera resolution (width)')
__parser.add_argument('--height', default=DEFAULT_HEIGHT, help='Camera resolution (height)')
__parser.add_argument('--frame_size', default=DEFAULT_FRAME_SIZE, help='Frame callback resolution (width and height)')

args = __parser.parse_args()
__debug: bool = args.debug
__disable_assertions: bool = args.disable_assertions


def is_debug() -> bool:
    return __debug


if __debug:
    def debug(message):
        print(message())
else:
    def debug(message):
        pass

if __disable_assertions:
    def expect(condition, message):
        pass
else:
    def expect(condition, message):
        assert condition(), message()
