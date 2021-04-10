import abc
from threading import Thread
from typing import Optional
from ui.state import State


ImageFrame = tuple[Optional[object], int]
EMPTY: ImageFrame = (None, 0)


class ImageSource(metaclass=abc.ABCMeta):
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


class VideoImageSource(ImageSource):

    __state: State = State.UNINITIALISED
    __thread: Optional[Thread] = None
    __image: ImageFrame = EMPTY
    __last: int = 0

    def __init__(self, camera, time, delay):
        self.__camera = camera
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
        image = frame if ok else None
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
