import cv2
from typing import Optional
from collections import deque
from tkinter import *
from PIL import Image, ImageTk
from timing.time_source import TimeSource
from ui.callback.callback import FrameCallback
from ui.source.image_source import ImageSource, VideoImageSource
from ui.state import State


class GUI:
    # TODO: resizing
    # TODO: hooking for drawing / classification / segmentation / etc

    __time = TimeSource()
    __state: State = State.UNINITIALISED
    __root = None
    __callback: FrameCallback
    __source = Optional[ImageSource]
    __delay_ms: int = 10
    __last: int = -1

    def __init__(self, title: str, width: int, height: int, callback: Optional[FrameCallback] = None, port: int = 0, history: int = 5):
        self.__title = title
        self.__width = width
        self.__height = height
        self.__callback = callback if callback is not None else FrameCallback(lambda frame: Image.fromarray(frame))
        self.__port = port
        self.__frames = deque([0.0] * history)

    @property
    def time(self):
        return self.__time

    @property
    def width(self):
        return int(self.__source.camera.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self.__source.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def title(self):
        return self.__title

    def start(self):
        if self.__state != State.UNINITIALISED:
            return
        self.__state = State.INTERMEDIATE
        self.__setup()
        self.__state = State.RUNNING
        self.__root.mainloop()

    def stop(self):
        if self.__state != State.RUNNING:
            return
        self.__state = State.INTERMEDIATE
        self.__destroy()
        self.__state = State.UNINITIALISED

    def __update_image(self, canvas):
        frame, timestamp = self.__source.image

        if frame is None or self.__last >= timestamp:
            return

        self.__last = timestamp

        image = self.__callback.invoke(frame)
        photo = ImageTk.PhotoImage(image=image)

        canvas.configure(image=photo)
        canvas.__cached = photo  # avoid garbage collection

    def __update_fps(self, canvas):
        frame_times = self.__frames
        frame_times.rotate()
        frame_times[0] = self.time.seconds

        sum_of_deltas = frame_times[0] - frame_times[-1]
        count_of_deltas = len(frame_times) - 1
        fps = 0 if sum_of_deltas == 0 else int(float(count_of_deltas) / sum_of_deltas)

        canvas.configure(text=f'FPS: {fps}')

    def __update_all(self, image, fps):
        root = self.__root
        self.__update_image(image)
        self.__update_fps(fps)
        root.update()
        root.after(self.__delay_ms, func=lambda: self.__update_all(image, fps))

    def __setup_source(self):
        self.__source = source = VideoImageSource(cv2.VideoCapture(self.__port), self.time, self.__delay_ms)
        source.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.__width)
        source.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.__height)
        source.start()

    def __setup_canvas(self):
        self.__root = root = Tk()
        root.wm_title(self.__title)

        # image component
        image_canvas = Label(master=root)
        image_canvas.pack()

        # FPS label
        fps_canvas = Label(master=root)
        fps_canvas.pack()

        # dimension info
        dimensions_canvas = Label(master=root, text=f'Dimensions: {self.width}x{self.height}px')
        dimensions_canvas.pack()

        # version info
        version_canvas = Label(master=root, text=f'TK/TCL: {TkVersion}/{TclVersion}')
        version_canvas.pack()

        # exit button
        exit_button = Button(master=root, text='Exit', command=lambda: self.__destroy())
        exit_button.pack()

        # setup the update callback
        root.after(0, func=lambda: self.__update_all(image_canvas, fps_canvas))

    def __setup(self):
        self.__setup_source()
        self.__setup_canvas()

    def __destroy_source(self):
        cv2.destroyAllWindows()
        self.__source.stop()
        self.source = None

    def __destroy_canvas(self):
        self.__root.destroy()
        self.__root = None

    def __destroy(self):
        self.__destroy_source()
        self.__destroy_canvas()
