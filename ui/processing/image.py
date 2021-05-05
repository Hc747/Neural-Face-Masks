import cv2
import numpy as np
from typing import Optional


def centered_crop(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_halved = int(np.amin([h, w]) / 2)
    h = int(h / 2)
    w = int(w / 2)

    cropped = img[h-min_halved:h+min_halved, w-min_halved:w+min_halved]
    return cv2.resize(cropped, (size, size), interpolation=interpolation)


def resize(frame, size, interpolation=cv2.INTER_AREA):
    return cv2.resize(frame, size, interpolation=interpolation)


def rescale(frame, scale: Optional[float], interpolation=cv2.INTER_AREA):
    if scale is None:
        return frame, frame, 1.0
    ratio: float = 1.0 / scale
    height, width = (np.asarray(np.shape(frame)[:2]) * ratio).astype(int)
    scaled = resize(frame, (width, height), interpolation)
    return frame, scaled, scale


def translate_scale(x, scale: float):
    return (np.asarray(x) * scale).astype(int)
