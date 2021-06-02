import cv2
from skimage import io

"""
A module providing functionality to validate the format/integrity of images.
"""

__functions = [cv2.imread, io.imread]


def is_valid_image(path) -> bool:
    """
    Determines whether or not an image at the path supplied is valid.
    :param path:
    The file system path specifying the location of the image.
    :return:
    True if the image is valid, False otherwise.
    """
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
                # access a property of the image to ensure it loaded correctly.
            except:
                return False
    return True
