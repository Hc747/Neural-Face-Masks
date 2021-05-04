import argparse
import os
import sys
from distutils.util import strtobool
from constants import MASK_DETECTOR_ASHISH, FACE_DETECTOR_SVM

DEFAULT_TITLE: str = 'FTF: Face Mask Analyser'
DEFAULT_IMAGE_SOURCE: str = 'video'
DEFAULT_FACE_DETECTOR_PATH: str = os.path.abspath(os.path.join('.', 'models', 'face', 'mmod_human_face_detector.dat'))
DEFAULT_DETECTOR: str = FACE_DETECTOR_SVM
DEFAULT_MASK_DETECTOR: str = MASK_DETECTOR_ASHISH
DEFAULT_WIDTH: int = 1024
DEFAULT_HEIGHT: int = 720
DEFAULT_SCALE: float = 2.5
DEFAULT_FACE_SIZE: int = 224
DEFAULT_ENABLE_DEBUG: bool = True
DEFAULT_ENABLE_ASSERTIONS: bool = False
DEFAULT_EXPERIMENTAL_FEATURES: bool = False


def __boolean(v: str) -> bool:
    return bool(strtobool(v))


# TODO: type safety
# TODO: mutual exclusion
__parser = argparse.ArgumentParser()
__parser.add_argument('--source', default=DEFAULT_IMAGE_SOURCE, type=str, help='Image source (video)')
__parser.add_argument('--face_detector', default=DEFAULT_DETECTOR, type=str, help='Face detector (realtime or accurate)')
__parser.add_argument('--face_detector_path', default=DEFAULT_FACE_DETECTOR_PATH, type=str, help='Face detector models path')
__parser.add_argument('--mask_detector', default=DEFAULT_MASK_DETECTOR, type=str, help='Mask detector implementation')
__parser.add_argument('--debug', default=DEFAULT_ENABLE_DEBUG, type=__boolean, help='Print debug statements (True/False)')
__parser.add_argument('--enable_assertions', type=__boolean, default=DEFAULT_ENABLE_ASSERTIONS, help='Disable assertions at runtime (True/False)')
__parser.add_argument('--title', default=DEFAULT_TITLE, type=str, help='Application GUI title')
__parser.add_argument('--width', default=DEFAULT_WIDTH, type=int, help='Camera resolution (width)')
__parser.add_argument('--height', default=DEFAULT_HEIGHT, type=int, help='Camera resolution (height)')
__parser.add_argument('--scale', default=DEFAULT_SCALE, type=float, help='Frame size scaling (to make processing more computationall efficient)')
__parser.add_argument('--experimental', default=DEFAULT_EXPERIMENTAL_FEATURES, type=__boolean, help='Enable experimental features')

# publicly exported
args = __parser.parse_args()


__debug: bool = args.debug
__experimental: bool = args.experimental
__assertions_enabled: bool = args.enable_assertions


def is_debug() -> bool:
    return __debug


def is_experimental() -> bool:
    return __experimental


def is_assertions_enabled() -> bool:
    return __assertions_enabled


# export debug statement or no-op
if __debug:
    def debug(message, out=sys.stdout):
        print(message(), file=out)
else:
    def debug(message, out=None):
        pass

# export expect statement or no-op
if __assertions_enabled:
    def expect(condition, message):
        assert condition(), message()
else:
    def expect(condition, message):
        pass
