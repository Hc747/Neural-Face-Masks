from PIL.Image import Image


class FrameCallback:
    def __init__(self, fn: lambda f: Image):
        self.__fn = fn

    def invoke(self, frame) -> Image:
        return self.__fn(frame)
