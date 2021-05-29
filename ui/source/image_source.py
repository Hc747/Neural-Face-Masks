import abc
import cv2
from threading import Thread
from typing import Optional, Tuple
from PIL import Image
from timing.time_source import TimeSource
from ui.callback.callback import FrameCallback
from ui.state import State


ImageFrame = Tuple[Optional[Image.Image], int]
EMPTY: ImageFrame = (None, 0)
EPS = 1e-9


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

    @property
    def fps(self) -> float:
        return 0.0

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
    __time: TimeSource
    __last: int = 0

    def __init__(self, camera, callback: FrameCallback, time: TimeSource, history: int = 10):
        self.__camera = camera
        self.__callback = callback
        self.__time = time
        self.__history = history
        self.__timing = [0.0] * history
        self.__timing_index = 0

    @property
    def image(self) -> ImageFrame:
        return self.__image

    @property
    def camera(self):
        return self.__camera

    @property
    def fps(self) -> float:
        return self.__history / (sum(self.__timing) + EPS)

    def __update(self):
        start = self.__time.millis
        ok, frame = self.camera.read()
        if ok:
            if self.raw or self.__last == 0:
                image = ImageSource.process_raw(frame)
            else:
                try:
                    image = self.__callback.invoke(frame)
                except Exception as e:
                    image = ImageSource.process_raw(frame)
                    print(f'Exception: {e}')
        else:
            image = None
        finish = self.__last = self.__time.millis
        duration = (finish - start) / 1_000
        self.__timing[self.__timing_index] = duration
        self.__timing_index = (self.__timing_index + 1) % self.__history
        self.__image = (image, finish)

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
        self.__thread = thread = Thread(target=self.__run, daemon=True)
        self.__state = State.RUNNING
        thread.start()

    def stop(self):
        if self.__state != State.RUNNING:
            return
        self.__state = State.INTERMEDIATE
        self.__thread.join()
