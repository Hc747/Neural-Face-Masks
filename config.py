import argparse
from distutils.util import strtobool
from constants import *

"""
Provides the runtime settings for the application.
"""

# The default runtime setting values.
DEFAULT_TITLE: str = 'FTF: Face Mask Analyser'
DEFAULT_DETECTOR: str = FACE_DETECTOR_MEDIA_PIPE
DEFAULT_MASK_DETECTOR: str = MASK_DETECTOR_ASHISH
DEFAULT_WIDTH: int = 1024
DEFAULT_HEIGHT: int = 720
DEFAULT_SCALE: float = 2.6
DEFAULT_FACE_SIZE: int = 224
DEFAULT_CACHE_FRAMES: int = 15
DEFAULT_PADDING: int = 65
DEFAULT_ENABLE_DEBUG: bool = False
DEFAULT_ENABLE_ASSERTIONS: bool = False
DEFAULT_EXPERIMENTAL_FEATURES: bool = False
DEFAULT_PRODUCTION: bool = True


def __boolean(v: str) -> bool:
    """
    A special type that'll parse true/false, yes/no, 1/0 as boolean values.
    """
    return bool(strtobool(v))


__parser = argparse.ArgumentParser()
__parser.add_argument('--cache_frames', default=DEFAULT_CACHE_FRAMES, type=int, help='Frame caching optimisation')
__parser.add_argument('--face_detector', default=DEFAULT_DETECTOR, type=str, help='Face detector (CNN, SVM or MediaPipe)')
__parser.add_argument('--mask_detector', default=DEFAULT_MASK_DETECTOR, type=str, help='Mask detector implementation')
__parser.add_argument('--debug', default=DEFAULT_ENABLE_DEBUG, type=__boolean, help='Print debug statements (True/False)')
__parser.add_argument('--enable_assertions',  default=DEFAULT_ENABLE_ASSERTIONS, type=__boolean, help='Disable assertions at runtime (True/False)')
__parser.add_argument('--title', default=DEFAULT_TITLE, type=str, help='Application GUI title')
__parser.add_argument('--width', default=DEFAULT_WIDTH, type=int, help='Camera resolution (width)')
__parser.add_argument('--height', default=DEFAULT_HEIGHT, type=int, help='Camera resolution (height)')
__parser.add_argument('--scale', default=DEFAULT_SCALE, type=float, help='Frame size scaling (to make processing more computationally efficient)')
__parser.add_argument('--padding', default=DEFAULT_PADDING, type=int, help='Face padding (to increase face boundary size)')
__parser.add_argument('--experimental', default=DEFAULT_EXPERIMENTAL_FEATURES, type=__boolean, help='Enable experimental features')
__parser.add_argument('--production', default=DEFAULT_PRODUCTION, type=__boolean, help='Run the application in production mode')

# publicly exported runtime settings
args = __parser.parse_args()
