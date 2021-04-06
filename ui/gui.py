# source: https://stackoverflow.com/questions/16366857/show-webcam-sequence-tkinter

from collections import deque
import cv2
from PIL import Image, ImageTk
import time
import tkinter as tk


def destroy(root):
    root.destroy()
    

def update_image(image_label, cam):
    ok, frame = cam.read()

    gray_im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    a = Image.fromarray(gray_im)
    b = ImageTk.PhotoImage(image=a)
    image_label.configure(image=b)

    image_label._image_cache = b  # avoid garbage collection
    root.update()


def update_fps(fps_label):
    frame_times = fps_label._frame_times
    frame_times.rotate()
    frame_times[0] = time.time()
    sum_of_deltas = frame_times[0] - frame_times[-1]
    count_of_deltas = len(frame_times) - 1
    fps = 0 if sum_of_deltas == 0 else int(float(count_of_deltas) / sum_of_deltas)
    fps_label.configure(text=f'FPS: {fps}')


def update_all(root, image_label, cam, fps_label):
    update_image(image_label, cam)
    update_fps(fps_label)
    root.after(20, func=lambda: update_all(root, image_label, cam, fps_label))


if __name__ == '__main__':
    root = tk.Tk()
    image_label = tk.Label(master=root) # label for the video frame
    image_label.pack()

    width, height = 320, 320
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fps_label = tk.Label(master=root) # label for fps
    fps_label._frame_times = deque([0]*5)  # arbitrary 5 frame average FPS
    fps_label.pack()
    # quit button
    quit_button = tk.Button(master=root, text='Quit', command=lambda: destroy(root))
    quit_button.pack()
    # setup the update callback
    root.after(0, func=lambda: update_all(root, image_label, cam, fps_label))
    root.mainloop()