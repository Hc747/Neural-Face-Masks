import abc
from abc import ABC

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
    'resnet50': ResNet50,
    'resnet50v2': ResNet50V2,
    'resnet101': ResNet101,
    'resnet101v2': ResNet101V2,
    'resnet152': ResNet152,
    'resnet152v2': ResNet152V2,
    'inceptionv3': InceptionV3
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


class StandardNetworkArchitecture(NetworkArchitecture, ABC):
    def __init__(self, shape, classes):
        self.__shape = shape
        self.__classes = classes


class StandardVGGNetworkArchitecture(StandardNetworkArchitecture):
    def __init__(self, shape, classes, vgg):
        super().__init__(shape, classes)
        self.__vgg = vgg

    def model(self):
        base = self.__vgg(weights='imagenet', include_top=False, input_shape=self.__shape, classes=self.__classes)
        base.trainable = False

        head = Flatten(name='flatten')(base.output)
        classification = Dense(4096, activation='relu', name='fc1')(head)
        classification = Dense(4096, activation='relu', name='fc2')(classification)
        classification = Dense(self.__classes, activation='softmax', name='predictions')(classification)
        return Model(inputs=base.input, outputs=classification)


class StandardResnetNetworkArchitecture(StandardNetworkArchitecture):
    def __init__(self, shape, classes, resnet):
        super().__init__(shape, classes)
        self.__resnet = resnet

    def model(self):
        base = self.__resnet(weights='imagenet', include_top=False, input_shape=self.__shape, classes=self.__classes)
        base.trainable = False

        head = GlobalAveragePooling2D(name='avg_pool')(base.output)
        classification = Dense(self.__classes, activation='softmax', name='predictions')(head)
        return Model(inputs=base.input, outputs=classification)


class StandardInceptionNetworkArchitecture(StandardNetworkArchitecture):
    def __init__(self, shape, classes, inception):
        super().__init__(shape, classes)
        self.__inception = inception

    def model(self):
        base = self.__inception(weights='imagenet', include_top=False, input_shape=self.__shape, classes=self.__classes)
        base.trainable = False

        head = GlobalAveragePooling2D(name='avg_pool')(base.output)
        classification = Dense(self.__classes, activation='softmax', name='predictions')(head)
        return Model(inputs=base.input, outputs=classification)


class ClassificationNetwork(NetworkArchitecture):
    __VGG_NETWORKS = [VGG16, VGG19]
    __RESNET_NETWORKS = [ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2]
    __INCEPTION_NETWORKS = [InceptionV3]

    def __init__(self, base, shape, classes):
        self.base = base
        self.__shape = shape
        self.__classes = classes

    @staticmethod
    def standard_architecture(network, shape, classes):
        if network in ClassificationNetwork.__VGG_NETWORKS:
            return StandardVGGNetworkArchitecture(shape, classes, vgg=network)
        elif network in ClassificationNetwork.__RESNET_NETWORKS:
            return StandardResnetNetworkArchitecture(shape, classes, resnet=network)
        elif network in ClassificationNetwork.__INCEPTION_NETWORKS:
            return StandardInceptionNetworkArchitecture(shape, classes, inception=network)
        else:
            raise ValueError(f'Unknown base classification network: {network}')

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
