import cv2
import numpy as np
from app.processing.image import resize
from constants import *
from detectors.mask.detectors import MaskDetector

"""
A module exporting utility functions for rendering on top of an image.
"""


def draw_boxes(frame, face_coordinates, face_colour, face_label, boundary_coordinates, boundary_colour, boundary_label):
    """
    Renders the detection boxes and labels upon the frame.
    """
    if np.allclose(face_coordinates[:2], boundary_coordinates[:2], atol=TEXT_OFFSET // 2):
        offset = int(-TEXT_OFFSET * 1.5)
    else:
        offset = TEXT_OFFSET
    # face
    cv2.rectangle(frame, face_coordinates[:2], face_coordinates[2:], face_colour, 1)
    cv2.putText(frame, text=face_label, org=(face_coordinates[0], face_coordinates[1] - TEXT_OFFSET), fontFace=FONT_FACE, fontScale=FONT_SCALE, color=face_colour)
    # boundary
    cv2.rectangle(frame, boundary_coordinates[:2], boundary_coordinates[2:], boundary_colour, 1)
    cv2.putText(frame, text=boundary_label, org=(boundary_coordinates[0], boundary_coordinates[1] - offset), fontFace=FONT_FACE, fontScale=FONT_SCALE, color=boundary_colour)


def draw_stats(frame, mask: MaskDetector, stats):
    """
    Renders the detection statistics upon the frame.
    """
    beginning, increment = 10, 15
    for (index, mapping) in enumerate(mask.mapping.values()):
        label = mapping.get('label')
        colour = mapping.get('colour')
        count: int = stats.get(label, 0)
        offset: int = beginning + (index * increment)
        cv2.putText(frame, f'{label}: {count}', org=(0, offset), fontFace=FONT_FACE, fontScale=FONT_SCALE, color=colour)


def draw_floating_head(frame, head, colour, index: int, items: int, size: int, height_offset: int, width_offset: int):
    """
    Renders the input 'floating heads' upon a frame.
    """
    # TODO: padding between images?
    row: int = int(index / items)
    column: int = int(index - (row * items))

    top: int = height_offset + (column * size)
    left: int = width_offset + (row * size)
    bottom, right = top + size, left + size

    image = resize(head, (size, size))
    frame[top:bottom, left:right] = image
    cv2.rectangle(frame, (left, top), (right, bottom), colour, 1)


def display_confidence(confidence):
    """
    Formats a prediction confidence value for display upon a frame.
    """
    return 'unknown' if confidence is None else f'{confidence:.02f}%'
