import cv2
from typing import Optional
from tkinter import *
from PIL import Image, ImageTk
from detectors.face.detectors import FaceDetectorProvider
from detectors.mask.detectors import MaskDetectorProvider
from timing.time_source import TimeSource
from ui.callback.callback import FrameCallback
from ui.source.image_source import ImageSource, VideoImageSource
from ui.state import State


class GUI:
    # TODO: resizing

    __time = TimeSource()
    __state: State = State.UNINITIALISED
    __root = None
    __callback: FrameCallback
    __source = Optional[ImageSource]
    __delay_ms: int = 10
    __last: int = -1

    def __init__(self, title: str, width: int, height: int, callback: Optional[FrameCallback] = None, port: int = 0):
        self.__title = title
        self.__width = width
        self.__height = height
        self.__callback = callback if callback is not None else FrameCallback(lambda frame: Image.fromarray(frame))
        self.__port = port

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

    def __update_image(self, canvas) -> bool:
        image, timestamp = self.__source.image

        if image is None or self.__last >= timestamp:
            return False

        self.__last = timestamp

        photo = ImageTk.PhotoImage(image=image)

        canvas.configure(image=photo)
        canvas.__cached = photo  # avoid garbage collection
        return True

    def __update_all(self, image):
        root = self.__root
        updated = self.__update_image(image)
        if updated:
            root.update()
        root.after(self.__delay_ms, func=lambda: self.__update_all(image))

    def __setup_image_source(self):
        self.__source = source = VideoImageSource(cv2.VideoCapture(self.__port), self.__callback, self.time, self.__delay_ms)
        source.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.__width)
        source.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.__height)
        source.start()

    def __setup_canvas(self):
        self.__root = root = Tk()
        root.wm_geometry(f'{root.winfo_screenwidth()}x{root.winfo_screenheight()}')
        root.wm_title(self.__title)

        # controls container
        controls_container = Frame(master=root)
        controls_container.pack()

        # image container
        image_container = Frame(master=root)
        image_container.pack()

        # info container
        info_container = Frame(master=root)
        info_container.pack()

        # image component
        image_canvas = Label(master=image_container)
        image_canvas.pack()

        # camera resolution
        resolution_canvas = Label(master=info_container, text=f'Native resolution: {self.width}x{self.height}px')
        resolution_canvas.pack()

        # gui version info
        gui_version_canvas = Label(master=info_container, text=f'TK/TCL: {TkVersion}/{TclVersion}')
        gui_version_canvas.pack()

        # api version info
        api_version_canvas = Label(master=info_container, text=f'{FaceDetectorProvider.version()}\n{MaskDetectorProvider.version()}')
        api_version_canvas.pack()

        # toggle button
        toggle_button = Button(master=controls_container, text='Toggle', command=lambda: self.__source.toggle_raw())
        toggle_button.pack(side=LEFT)

        # exit button
        exit_button = Button(master=controls_container, text='Exit', command=lambda: self.__destroy())
        exit_button.pack(side=RIGHT)

        # setup the update callback
        root.after(0, func=lambda: self.__update_all(image_canvas))

    def __setup(self):
        self.__setup_image_source()
        self.__setup_canvas()

    def __destroy_image_source(self):
        cv2.destroyAllWindows()
        self.__source.stop()
        self.source = None

    def __destroy_canvas(self):
        self.__root.destroy()
        self.__root = None

    def __destroy(self):
        self.__destroy_image_source()
        self.__destroy_canvas()
