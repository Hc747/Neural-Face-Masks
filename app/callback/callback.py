import abc
from PIL.Image import Image

"""
A module exporting interfaces related to the processing of images / video frames.
"""


class FrameCallback(metaclass=abc.ABCMeta):
    """
    An interface defining the contract for converting a CV2 frame into a Pillow Image.
    """

    @abc.abstractmethod
    def invoke(self, frame) -> Image:
        """
        An abstract method for performing some form of processing on an Image to be rendered.
        """
        raise ValueError('FrameCallback#Invoke has not been implemented.')


class LambdaFrameCallback(FrameCallback):
    """
    An implementation of the FrameCallback that allows for the use of a parameterised lambda function in place of
    a concrete implementation/subclass of FrameCallback. Useful for prototyping, etc.
    """
    def __init__(self, fn):
        self.__fn = fn

    def invoke(self, frame) -> Image:
        return self.__fn(frame)
