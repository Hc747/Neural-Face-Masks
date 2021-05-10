import cv2
from typing import Optional
from tkinter import *
from PIL import Image, ImageTk
from configuration.configuration import ApplicationConfiguration
from constants import FACE_DETECTOR_SVM, FACE_DETECTOR_CNN, MIN_SCALE, MAX_SCALE
from detectors.face.detectors import FaceDetectorProvider
from detectors.mask.detectors import MaskDetectorProvider
from timing.time_source import TimeSource
from ui.callback.callback import FrameCallback, LambdaFrameCallback
from ui.source.image_source import ImageSource, VideoImageSource
from ui.state import State


class GUI:
    __time = TimeSource()
    __state: State = State.UNINITIALISED
    __root = None
    __configuration: ApplicationConfiguration
    __callback: FrameCallback
    __source = Optional[ImageSource]
    __delay_ms: int = 10
    __last: int = -1

    def __init__(self, title: str, width: int, height: int, configuration: ApplicationConfiguration, callback: Optional[FrameCallback] = None, port: int = 0):
        self.__title = title
        self.__width = width
        self.__height = height
        self.__configuration = configuration
        self.__callback = callback if callback is not None else LambdaFrameCallback(lambda frame: Image.fromarray(frame))
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
        canvas.cached = photo  # avoid garbage collection
        return True

    def __update_fps(self, fps) -> bool:
        fps.configure(text=f'FPS: {self.__source.fps:.2f}')
        return True

    def __update_all(self, image, fps):
        root = self.__root
        # bitwise 'and' intentional in order to allow fallthrough evaluation
        updated = self.__update_image(image) & self.__update_fps(fps)
        if updated:
            root.update()
        root.after(self.__delay_ms, func=lambda: self.__update_all(image, fps))

    def __setup_image_source(self):
        self.__source = source = VideoImageSource(cv2.VideoCapture(self.__port), self.__callback, self.time)
        source.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.__width)
        source.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.__height)
        source.start()

    def __setup_canvas(self):
        self.__root = root = Tk()
        root.wm_geometry(f'{root.winfo_screenwidth()}x{root.winfo_screenheight()}')
        root.wm_title(self.__title)

        # image container
        images = Frame(master=root)
        images.pack(anchor=W, side=LEFT)

        # image component
        canvas = Label(master=images)
        canvas.pack()

        # label and control container
        components = Frame(master=root)
        components.pack(anchor=W, side=LEFT)

        # controls container
        controls_container = Frame(master=components)
        controls_container.pack(anchor=W)

        # info container
        info_container = Frame(master=components)
        info_container.pack(anchor=W)

        detector = StringVar(master=controls_container, value=self.__configuration.face_str(), name='detector')

        def update_detector():
            self.__configuration.face = detector.get()

        # FPS
        fps = Label(master=info_container, text='FPS')

        if self.__configuration.production:
            controls = [
                # exit button
                Button(master=controls_container, text='Exit', command=lambda: self.__destroy()),
                # toggle button
                Checkbutton(master=controls_container, text='Show Raw', command=lambda: self.__source.toggle_raw()),
                # SVM face detector
                Radiobutton(master=controls_container, text='Higher FPS', value=FACE_DETECTOR_SVM, variable=detector, command=update_detector),
                # CNN face detector
                Radiobutton(master=controls_container, text='Higher Accuracy', value=FACE_DETECTOR_CNN, variable=detector, command=update_detector)
            ]
            info = [
                # FPS
                fps,
                # resolution
                Label(master=info_container, text=f'{self.width}x{self.height}px'),
            ]
        else:
            controls = [
                # exit button
                Button(master=controls_container, text='Exit', command=lambda: self.__destroy()),
                # debugging
                Checkbutton(master=controls_container, text='Debug', command=lambda: self.__configuration.toggle_debugging()),
                # asserting
                Checkbutton(master=controls_container, text='Assert', command=lambda: self.__configuration.toggle_asserting()),
                # experimenting
                Checkbutton(master=controls_container, text='Experiment', command=lambda: self.__configuration.toggle_experimenting()),
                # toggle button
                Checkbutton(master=controls_container, text='Show Raw', command=lambda: self.__source.toggle_raw()),
                # SVM face detector
                Radiobutton(master=controls_container, text='Higher FPS', value=FACE_DETECTOR_SVM, variable=detector, command=update_detector),
                # CNN face detector
                Radiobutton(master=controls_container, text='Higher Accuracy', value=FACE_DETECTOR_CNN, variable=detector, command=update_detector)
            ]
            info = [
                # FPS
                fps,
                # resolution
                Label(master=info_container, text=f'{self.width}x{self.height}px'),
                # tk version
                Label(master=info_container, text=f'TK: {TkVersion}'),
                # tcl version
                Label(master=info_container, text=f'TCL: {TclVersion}')
            ]


        [control.pack(anchor=W) for control in controls]
        [display.pack(anchor=W) for display in info]

        # setup the update callback
        root.after(0, func=lambda: self.__update_all(canvas, fps))

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
