import cv2
from skimage import io

__functions = [cv2.imread, io.imread]


def is_valid_image(path) -> bool:
    for fn in __functions:
        try:
            image = fn(path)
        except:
            return False
        if image is None:
            return False
        else:
            try:
                shape = image.shape
            except:
                return False
    return True
