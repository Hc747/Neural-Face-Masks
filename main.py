import mediapipe as mp  # application deadlocks if this isn't imported before tensorflow
# import tensorflowjs as tfjs
from config import args
from configuration.configuration import ApplicationConfiguration, debug
from constants import *
from detectors.face.detectors import FaceDetectorProvider
from detectors.mask.detectors import MaskDetectorProvider
from ui.callback.application_callback import ApplicationCallback
from ui.callback.callback import FrameCallback
from ui.gui import GUI

# TODO: logging
# TODO: JIT compilation?
# TODO: add more classes
# TODO: relocate..?

if __name__ == '__main__':
    debug(lambda: f'Application configuration: {args}')
    debug(FaceDetectorProvider.version)
    debug(MaskDetectorProvider.version)

    with FaceDetectorProvider.get_face_detector(FACE_DETECTOR_MEDIA_PIPE, confidence=0.5) as mp_face:
        faces = {
            FACE_DETECTOR_MEDIA_PIPE: mp_face,
            FACE_DETECTOR_CNN: FaceDetectorProvider.get_face_detector(FACE_DETECTOR_CNN, filename=args.face_detector_path),
            FACE_DETECTOR_SVM: FaceDetectorProvider.get_face_detector(FACE_DETECTOR_SVM)
        }

        mask = MaskDetectorProvider.get_mask_detector(args.mask_detector)  # TODO: typing

        # if args.dump_js:
        #     tfjs.converters.save_keras_model(mask, './electron/app/src/model')

        configuration: ApplicationConfiguration = ApplicationConfiguration(args, faces=faces, mask=mask)
        callback: FrameCallback = ApplicationCallback(configuration)

        gui = GUI(title=args.title, width=args.width, height=args.height, configuration=configuration, callback=callback)
        gui.start()
