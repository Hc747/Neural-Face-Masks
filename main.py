# noinspection PyUnresolvedReferences
import mediapipe
# application deadlocks if this isn't imported before tensorflow
import os
from app.configuration.configuration import ApplicationConfiguration, debug
from app.callback.application_callback import ApplicationCallback
from app.callback.callback import FrameCallback
from app.gui import GUI
from constants import *
from config import args
from detectors.face.detectors import FaceDetectorProvider
from detectors.mask.detectors import MaskDetectorProvider

"""
The application entrypoint.
"""

if __name__ == '__main__':
    debug(lambda: f'Application configuration: {args}')
    debug(FaceDetectorProvider.version)
    debug(MaskDetectorProvider.version)

    confidence: float = args.confidence
    filename: str = os.path.abspath(os.path.join('.', 'models', 'face', 'dlib', 'mmod_human_face_detector.dat'))

    # MediaPipe face detector needs to manage resources properly; therefore used in with block.
    with FaceDetectorProvider.get_face_detector(FACE_DETECTOR_MEDIA_PIPE, confidence=confidence) as mp_face:
        # define face detectors
        faces = {
            FACE_DETECTOR_MEDIA_PIPE: mp_face,
            FACE_DETECTOR_CNN: FaceDetectorProvider.get_face_detector(FACE_DETECTOR_CNN, filename=filename),
            FACE_DETECTOR_SVM: FaceDetectorProvider.get_face_detector(FACE_DETECTOR_SVM)
        }

        # define mask detectors
        masks = {
            MASK_DETECTOR_CABANI: MaskDetectorProvider.cabani(),
            MASK_DETECTOR_ASHISH: MaskDetectorProvider.ashish()
        }

        # define application configuration
        configuration: ApplicationConfiguration = ApplicationConfiguration(args, faces=faces, masks=masks)

        # define frame callback (application processing logic)
        callback: FrameCallback = ApplicationCallback(configuration)

        title, width, height = args.title, args.width, args.height

        # start the GUI
        gui = GUI(title=title, width=width, height=height, configuration=configuration, callback=callback)
        gui.start()
