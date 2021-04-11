import abc
import cv2
from threading import Thread
from typing import Optional
from PIL import Image
from ui.callback.callback import FrameCallback
from ui.state import State


ImageFrame = tuple[Optional[Image.Image], int]
EMPTY: ImageFrame = (None, 0)


class ImageSource(metaclass=abc.ABCMeta):

    __raw: bool = False

    @property
    @abc.abstractmethod
    def image(self) -> ImageFrame:
        pass

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

    @property
    def raw(self):
        return self.__raw

    def toggle_raw(self):
        self.__raw = not self.__raw

    @staticmethod
    def process_raw(frame) -> Image.Image:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)


class VideoImageSource(ImageSource):

    __state: State = State.UNINITIALISED
    __thread: Optional[Thread] = None
    __image: ImageFrame = EMPTY
    __callback: FrameCallback
    __last: int = 0

    def __init__(self, camera, callback: FrameCallback, time, delay):
        self.__camera = camera
        self.__callback = callback
        self.__time = time
        self.__delay = delay

    @property
    def image(self) -> ImageFrame:
        return self.__image

    @property
    def camera(self):
        return self.__camera

    def __update(self):
        now = self.__time.millis
        if (now - self.__last) < self.__delay:
            return
        self.__last = now

        ok, frame = self.camera.read()

        if ok:
            image = ImageSource.process_raw(frame) if self.raw else self.__callback.invoke(frame)
        else:
            image = None
        self.__image = (image, now)

    def __run(self):
        while self.__state == State.RUNNING:
            self.__update()
        self.__cleanup()

    def __cleanup(self):
        self.__thread = None
        self.__image = EMPTY
        self.__camera.release()
        self.__camera = None
        self.__state = State.UNINITIALISED

    def start(self):
        if self.__state != State.UNINITIALISED:
            return
        self.__state = State.INTERMEDIATE
        self.__thread = thread = Thread(target=self.__run)
        thread.daemon = True
        self.__state = State.RUNNING
        thread.start()

    def stop(self):
        if self.__state != State.RUNNING:
            return
        self.__state = State.INTERMEDIATE
        self.__thread.join()
