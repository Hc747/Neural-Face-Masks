import cv2
import numpy as np
from constants import *


def draw_boxes(frame, face_coordinates, face_colour, face_label, boundary_coordinates, boundary_colour, boundary_label):
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


def draw_stats(frame, masked: int, unmasked: int):
    cv2.putText(frame, f'Masked: {masked}', org=(0, 10), fontFace=FONT_FACE, fontScale=FONT_SCALE, color=COLOUR_GREEN)
    cv2.putText(frame, f'Unmasked: {unmasked}', org=(0, 25), fontFace=FONT_FACE, fontScale=FONT_SCALE, color=COLOUR_RED)
