import abc
from tensorflow.keras import Model
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *

BOUNDARY_NETWORK_NAME: str = 'boundary'
CLASSIFICATION_NETWORK_NAME: str = 'classification'

LOSS_FUNCTIONS = {
    BOUNDARY_NETWORK_NAME: 'mean_squared_error', # TODO: L2
    CLASSIFICATION_NETWORK_NAME: 'categorical_crossentropy'
}

LOSS_WEIGHTS = {
    BOUNDARY_NETWORK_NAME: 1.0,
    CLASSIFICATION_NETWORK_NAME: 1.0
}

NETWORKS = {
    'vgg16': VGG16,
    'vgg19': VGG19,
    'resnet': ResNet50
}


class NetworkArchitecture(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def model(self):
        raise NotImplementedError('NetworkArchitecture#model has not been implemented.')

    def compile(self, loss, weights, optimiser='adam', metrics=['accuracy']):
        return NetworkArchitecture.compile_static(self.model(), loss, weights, optimiser, metrics)

    @staticmethod
    def compile_static(model, loss, weights, optimiser='adam', metrics=['accuracy']):
        model.compile(optimizer=optimiser, metrics=metrics, loss=loss, loss_weights=weights)
        return model


class BaseVGG16ClassificationNetwork(NetworkArchitecture):
    def __init__(self, shape, classes):
        self.__shape = shape
        self.__classes = classes

    def model(self):
        base = VGG16(weights='imagenet', include_top=False, input_shape=self.__shape, classes=self.__classes)
        base.trainable = False

        head = Flatten(name='flatten')(base.output)
        classification = Dense(4096, activation='relu', name='fc1')(head)
        classification = Dense(4096, activation='relu', name='fc2')(classification)
        classification = Dense(self.__classes, activation='softmax', name='predictions')(classification)
        return Model(inputs=base.input, outputs=classification)


class ClassificationNetwork(NetworkArchitecture):
    def __init__(self, base, shape, classes):
        self.base = base
        self.__shape = shape
        self.__classes = classes

    @staticmethod
    def unmodified_base(base, shape, classes):
        if base == VGG16:
            return BaseVGG16ClassificationNetwork(shape, classes)
        else:
            raise ValueError(f'Unknown base classification network: {base}')

    def model(self):
        base = self.base(weights='imagenet', include_top=False, input_shape=self.__shape)
        base.trainable = False

        head = Flatten()(base.output)
        classification = Dense(512, activation='relu')(head)
        classification = Dropout(0.5)(classification)
        classification = Dense(512, activation='relu')(classification)
        classification = Dropout(0.5)(classification)
        classification = Dense(len(self.__classes), activation='softmax', name=CLASSIFICATION_NETWORK_NAME)(classification)
        return Model(inputs=base.input, outputs=classification)


class ClassificationRegressionNetwork(NetworkArchitecture):
    def __init__(self, base, shape, classes):
        self.base = base
        self.__shape = shape
        self.__classes = classes

    def model(self):
        base = self.base(weights='imagenet', include_top=False, input_shape=self.__shape)
        base.trainable = False

        head = Flatten()(base.output)

        # TODO: multiple detections per image
        boundary = Dense(256, activation='relu')(head)
        boundary = Dense(128, activation='relu')(boundary)
        boundary = Dense(64, activation='relu')(boundary)
        boundary = Dense(32, activation='relu')(boundary)
        boundary = Dense(4, activation='sigmoid', name=BOUNDARY_NETWORK_NAME)(boundary)
        # TODO: ensure input is scaled to 0,1 (x/255.0)

        # TODO: multiple classifications per image
        classification = Dense(512, activation='relu')(head)
        classification = Dropout(0.5)(classification)
        classification = Dense(512, activation='relu')(classification)
        classification = Dropout(0.5)(classification)
        classification = Dense(len(self.__classes), activation='softmax', name=CLASSIFICATION_NETWORK_NAME)(classification)
        return Model(inputs=base.input, outputs=(boundary, classification))
