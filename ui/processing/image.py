import cv2
import numpy as np


def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_halved = int(np.amin([h, w]) / 2)
    h = int(h / 2)
    w = int(w / 2)

    cropped = img[h-min_halved:h+min_halved, w-min_halved:w+min_halved]
    return cv2.resize(cropped, (size, size), interpolation=interpolation)
