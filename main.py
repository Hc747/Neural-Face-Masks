from keras.models import Model  # TODO: return layer of abstraction
from config import args
from configuration.configuration import ApplicationConfiguration, debug
from constants import *
from detectors.face.detectors import FaceDetectorProvider, FaceDetector
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

    svm: FaceDetector = FaceDetectorProvider.get_face_detector(FACE_DETECTOR_SVM)
    cnn: FaceDetector = FaceDetectorProvider.get_face_detector(FACE_DETECTOR_CNN, filename=args.face_detector_path)
    mask: Model = MaskDetectorProvider.get_mask_detector(args.mask_detector)

    configuration: ApplicationConfiguration = ApplicationConfiguration(args, svm=svm, cnn=cnn, mask=mask)
    callback: FrameCallback = ApplicationCallback(configuration)

    gui = GUI(title=args.title, width=args.width, height=args.height, configuration=configuration, callback=callback)
    gui.start()
