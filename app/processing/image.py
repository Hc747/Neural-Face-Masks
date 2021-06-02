import cv2
import numpy as np
from typing import Optional, Tuple

"""
A module exporting utility functions for image processing.
"""


def centered_crop(img, size, interpolation=cv2.INTER_AREA):
    """
    Crops an image into a square of the specified size originating from the center.
    """
    h, w = img.shape[:2]
    min_halved = int(np.amin([h, w]) / 2)
    h = int(h / 2)
    w = int(w / 2)

    cropped = img[h-min_halved:h+min_halved, w-min_halved:w+min_halved]
    return cv2.resize(cropped, (size, size), interpolation=interpolation)


def resize(frame, size, interpolation=cv2.INTER_AREA):
    """
    Resizes an image to the specified size.
    """
    return cv2.resize(frame, size, interpolation=interpolation)


def rescale(frame, scale: Optional[float], interpolation=cv2.INTER_AREA):
    """
    Rescales an image by the specified scale.
    """
    if scale is None:
        return frame, frame, 1.0
    ratio: float = 1.0 / scale
    height, width = (np.asarray(np.shape(frame)[:2]) * ratio).astype(int)
    scaled = resize(frame, (width, height), interpolation)
    return frame, scaled, scale


def translate_scale(x, scale: float):
    """
    Translates coordinates (x) from one scale to the specified scale.
    """
    return (np.asarray(x) * scale).astype(int)


def adjust_bounding_box(left: int, top: int, right: int, bottom: int, offset: int) -> Tuple[int, int, int, int, int, int]:
    """
    Applies padding to the coordinates of a bounding box.
    """
    left, top = left - offset, top - offset
    right, bottom = right + offset, bottom + offset
    width, height = right - left, bottom - top
    return left, top, right, bottom, width, height
