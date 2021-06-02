import abc
from PIL.Image import Image


class FrameCallback(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def invoke(self, frame) -> Image:
        raise ValueError('FrameCallback#Invoke has not been implemented.')


class LambdaFrameCallback(FrameCallback):
    def __init__(self, fn):
        self.__fn = fn

    def invoke(self, frame) -> Image:
        return self.__fn(frame)
