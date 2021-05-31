import cv2
from typing import Optional
from tkinter import *
from PIL import Image, ImageTk
from configuration.configuration import ApplicationConfiguration
from constants import FACE_DETECTOR_SVM, FACE_DETECTOR_CNN, FACE_DETECTOR_MEDIA_PIPE, MASK_DETECTOR_CABANI, \
    MASK_DETECTOR_ASHISH
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
    __delay_ms: int = 0
    __history: int = 30
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
        self.__source = source = VideoImageSource(cv2.VideoCapture(self.__port), self.__callback, self.time, history=self.__history)
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

        # FPS
        fps = Label(master=info_container, text='FPS')

        if self.__configuration.production:
            controls = [
                # toggle button
                Checkbutton(master=controls_container, text='Show Raw', command=lambda: self.__source.toggle_raw())
            ]
            info = [
                # FPS
                fps,
                # resolution
                Label(master=info_container, text=f'{self.width}x{self.height}px'),
            ]
        else:
            controls = [
                # debugging
                Checkbutton(master=controls_container, text='Debug', command=lambda: self.__configuration.toggle_debugging()),
                # asserting
                Checkbutton(master=controls_container, text='Assert', command=lambda: self.__configuration.toggle_asserting()),
                # experimenting
                Checkbutton(master=controls_container, text='Experiment', command=lambda: self.__configuration.toggle_experimenting()),
                # toggle button
                Checkbutton(master=controls_container, text='Show Raw', command=lambda: self.__source.toggle_raw())
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

        def update_face_detector():
            self.__configuration.face = face_detector.get()

        def update_mask_detector():
            self.__configuration.mask = mask_detector.get()

        face_detectors_container = Frame(master=controls_container)
        face_detector = StringVar(master=face_detectors_container, value=self.__configuration.face_str(), name='face_detector')

        face_detectors_label = Label(master=face_detectors_container, text='Face Config')
        face_svm = Radiobutton(master=face_detectors_container, text='Higher FPS', value=FACE_DETECTOR_SVM, variable=face_detector, command=update_face_detector)
        face_media_pipe = Radiobutton(master=face_detectors_container, text='Balanced', value=FACE_DETECTOR_MEDIA_PIPE, variable=face_detector, command=update_face_detector)
        face_cnn = Radiobutton(master=face_detectors_container, text='Higher Accuracy', value=FACE_DETECTOR_CNN, variable=face_detector, command=update_face_detector)

        mask_detectors_container = Frame(master=controls_container)
        mask_detector = StringVar(master=mask_detectors_container, value=self.__configuration.mask_str(), name='mask_detector')

        mask_detectors_label = Label(master=mask_detectors_container, text='Mask Config')
        mask_5_classes = Radiobutton(master=mask_detectors_container, text='Complex', value=MASK_DETECTOR_CABANI, variable=mask_detector, command=update_mask_detector)
        mask_2_classes = Radiobutton(master=mask_detectors_container, text='Simple', value=MASK_DETECTOR_ASHISH, variable=mask_detector, command=update_mask_detector)

        def adjust_cache(label, value: int):
            self.__configuration.cache_frames += value
            label.configure(text=f'Refresh: {self.__configuration.cache_frames}')

        cache_container = Frame(master=controls_container)
        cache_label = Label(master=cache_container, text=f'Refresh: {self.__configuration.cache_frames}')
        cache_decrement = Button(master=cache_container, text='-', command=lambda: adjust_cache(cache_label, -1))
        cache_increment = Button(master=cache_container, text='+', command=lambda: adjust_cache(cache_label, 1))

        def adjust_scale(label, value: float):
            self.__configuration.scale += value
            label.configure(text=f'Downscale: {self.__configuration.scale:.1f}')

        scale_container = Frame(master=controls_container)
        scale_label = Label(master=scale_container, text=f'Downscale: {self.__configuration.scale:.1f}')
        scale_decrement = Button(master=scale_container, text='-', command=lambda: adjust_scale(scale_label, -0.1))
        scale_increment = Button(master=scale_container, text='+', command=lambda: adjust_scale(scale_label, 0.1))

        for container in [face_detectors_container, mask_detectors_container, cache_container, scale_container]:
            controls.append(container)

        face_controls = [face_detectors_label, face_svm, face_media_pipe, face_cnn]
        mask_controls = [mask_detectors_label, mask_5_classes, mask_2_classes]
        cache_controls = [cache_label, cache_increment, cache_decrement]
        scale_controls = [scale_label, scale_increment, scale_decrement]

        [control.pack(anchor=W) for control in controls]
        [display.pack(anchor=W) for display in info]
        [face.pack(anchor=W) for face in face_controls]
        [mask.pack(anchor=W) for mask in mask_controls]
        [cache.pack(anchor=W) for cache in cache_controls]
        [scale.pack(anchor=W) for scale in scale_controls]

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
