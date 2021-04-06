# source: https://stackoverflow.com/questions/16366857/show-webcam-sequence-tkinter
from enum import Enum
from collections import deque
from threading import Thread

import cv2
import time as timing
from tkinter import *
from PIL import Image, ImageTk


class TimeSource:
    @property
    def seconds(self):
        return timing.time()

    @property
    def millis(self):
        return int(self.seconds * 1000)

    @property
    def nanos(self):
        return timing.time_ns()

class State(Enum):
    UNINITIALISED = 0
    INTERMEDIATE = 1
    RUNNING = 2


class CameraSource:
    __state: State = State.UNINITIALISED
    __thread = None
    __image = (None, 0)
    __last = 0

    def __init__(self, camera, time, __delay_ms):
        self.__camera = camera
        self.__time = time
        self.__delay = __delay_ms

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
        self.__image = None
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

    @property
    def image(self):
        return self.__image

    @property
    def camera(self):
        return self.__camera

class GUI:
    # TODO: resizing
    # TODO: hooking for drawing / classification / segmentation / etc

    __time = TimeSource()
    __state: State = State.UNINITIALISED
    __root = None
    __source = None
    __delay_ms: int = 10
    __last: int = -1

    def __init__(self, title: str, width: int, height: int, port: int = 0, history: int = 5):
        self.__title = title
        self.__width = width
        self.__height = height
        self.__port = port
        self.__frames = deque([0] * history)

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

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # TODO: externalise...
        scale = 0.60
        w = int(frame.shape[1] * scale)
        h = int(frame.shape[0] * scale)
        d = (w, h)

        scaled = cv2.resize(grayscale, d, interpolation=cv2.INTER_CUBIC)

        array = Image.fromarray(scaled)
        image = ImageTk.PhotoImage(image=array)

        canvas.configure(image=image)
        canvas.__cached = image  # avoid garbage collection

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
        self.__source = source = CameraSource(cv2.VideoCapture(self.__port), self.time, self.__delay_ms)
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
        fps_component = Label(master=root)
        fps_component.pack()

        # dimension info
        dimension_component = Label(master=root, text=f'Dimensions: {self.width}x{self.height}px')
        dimension_component.pack()

        # version info
        version_component = Label(master=root, text=f'TK/TCL: {TkVersion}/{TclVersion}')
        version_component.pack()

        # exit button
        exit_button = Button(master=root, text='Exit', command=lambda: self.__destroy())
        exit_button.pack()

        # setup the update callback
        root.after(0, func=lambda: self.__update_all(image_canvas, fps_component))

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


if __name__ == '__main__':
    gui = GUI(title='Webcam', width=640, height=480)
    gui.start()
