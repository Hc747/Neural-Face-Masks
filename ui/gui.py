# source: https://stackoverflow.com/questions/16366857/show-webcam-sequence-tkinter
from enum import Enum
from collections import deque
import cv2
from PIL import Image, ImageTk
import time
import tkinter as tk


class GUI:
    # TODO: resizing
    # TODO: hooking for drawing / classification / segmentation / etc

    class State(Enum):
        UNINITIALISED = 0
        INTERMEDIATE = 1
        RUNNING = 2

    __state: State = State.UNINITIALISED
    __root = None
    __camera = None

    def __init__(self, title: str, width: int, height: int, port: int = 0, history: int = 5):
        self.__title = title
        self.__width = width
        self.__height = height
        self.__port = port
        self.__frames = deque([0] * history)

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def title(self):
        return self.__title

    def start(self):
        if self.__state != GUI.State.UNINITIALISED:
            return
        self.__state = GUI.State.INTERMEDIATE
        self.__setup()
        self.__state = GUI.State.RUNNING
        self.__root.mainloop()

    def stop(self):
        if self.__state != GUI.State.RUNNING:
            return
        self.__state = GUI.State.INTERMEDIATE
        self.__destroy()
        self.__state = GUI.State.UNINITIALISED

    def __update_image(self, canvas):
        ok, frame = self.__camera.read()

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        array = Image.fromarray(grayscale)
        image = ImageTk.PhotoImage(image=array)

        canvas.configure(image=image)
        canvas._image_cache = image  # avoid garbage collection

    def __update_fps(self, canvas):
        frame_times = self.__frames
        frame_times.rotate()
        frame_times[0] = time.time()

        sum_of_deltas = frame_times[0] - frame_times[-1]
        count_of_deltas = len(frame_times) - 1
        fps = 0 if sum_of_deltas == 0 else int(float(count_of_deltas) / sum_of_deltas)

        canvas.configure(text=f'FPS: {fps}')

    def __update_all(self, image, fps):
        self.__update_image(image)
        self.__update_fps(fps)
        self.__root.update()
        self.__root.after(20, func=lambda: self.__update_all(image, fps))

    def __setup_camera(self):
        self.__camera = camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.__width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.__height)

    def __setup_canvas(self):
        self.__root = root = tk.Tk()
        root.wm_title(self.__title)

        # image component
        image_canvas = tk.Label(master=root)
        image_canvas.pack()

        # FPS label
        fps_component = tk.Label(master=root)
        fps_component.pack()

        # exit button
        exit_button = tk.Button(master=root, text='Exit', command=lambda: self.__destroy())
        exit_button.pack()

        # setup the update callback
        root.after(0, func=lambda: self.__update_all(image_canvas, fps_component))

    def __setup(self):
        self.__setup_camera()
        self.__setup_canvas()

    def __destroy_camera(self):
        cv2.destroyAllWindows()
        self.__camera.release()
        self.__camera = None

    def __destroy_canvas(self):
        self.__root.destroy()
        self.__root = None

    def __destroy(self):
        self.__destroy_camera()
        self.__destroy_canvas()


if __name__ == '__main__':
    gui = GUI(title='Webcam', width=320, height=320)
    gui.start()