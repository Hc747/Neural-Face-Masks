import mediapipe as mp
# application deadlocks if this isn't imported before tensorflow
import os
from config import args
from configuration.configuration import ApplicationConfiguration, debug
from constants import *
from detectors.face.detectors import FaceDetectorProvider
from detectors.mask.detectors import MaskDetectorProvider, MaskDetector
from ui.callback.application_callback import ApplicationCallback
from ui.callback.callback import FrameCallback
from ui.gui import GUI

if __name__ == '__main__':
    debug(lambda: f'Application configuration: {args}')
    debug(FaceDetectorProvider.version)
    debug(MaskDetectorProvider.version)

    with FaceDetectorProvider.get_face_detector(FACE_DETECTOR_MEDIA_PIPE, confidence=0.5) as mp_face:
        cnn_path: str = os.path.abspath(os.path.join('.', 'models', 'face', 'dlib', 'mmod_human_face_detector.dat'))
        faces = {
            FACE_DETECTOR_MEDIA_PIPE: mp_face,
            FACE_DETECTOR_CNN: FaceDetectorProvider.get_face_detector(FACE_DETECTOR_CNN, filename=cnn_path),
            FACE_DETECTOR_SVM: FaceDetectorProvider.get_face_detector(FACE_DETECTOR_SVM)
        }

        masks = {
            MASK_DETECTOR_CABANI: MaskDetectorProvider.cabani(),
            MASK_DETECTOR_ASHISH: MaskDetectorProvider.ashish()
        }

        configuration: ApplicationConfiguration = ApplicationConfiguration(args, faces=faces, masks=masks)
        callback: FrameCallback = ApplicationCallback(configuration)

        gui = GUI(title=args.title, width=args.width, height=args.height, configuration=configuration, callback=callback)
        gui.start()
