import abc
from abc import ABC
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import glorot_uniform, Constant

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

kernel_init = glorot_uniform()
bias_init = Constant(value=0.2)


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
    def __init__(self, shape, classes, base, trainable=False):
        self.__shape = shape
        self.__classes = classes
        self.__base = base
        self.__trainable = trainable

    def model(self):
        classes = len(self.__classes)
        base = self.__base(weights='imagenet', include_top=True, input_shape=self.__shape)
        base.trainable = self.__trainable

        head = base.layers[-4].output
        x = layers.Conv2D(classes * 64, kernel_size=1, padding='same', name='Logits')(head)
        x = layers.Flatten()(x)
        classification = Dense(classes, activation='softmax')(x)
        return Model(inputs=base.input, outputs=classification)

        # base.layers.pop()

        # x = layers.GlobalMaxPooling2D(name='max_pool')(base.output)
        # x = layers.Conv2D(classes, kernel_size=1, padding='same', name='Logits')(x)
        # x = layers.Flatten()(x)
        # classification = Dense(classes, activation='softmax')(x)
        # return Model(inputs=base.input, outputs=classification)
        # return base
        # base.trainable = self.__trainable
        #
        # head = Flatten()(base.output)
        # classification = Dense(1024, activation='relu')(head)
        # # classification = Dropout(0.35)(classification)
        # classification = Dense(1024, activation='relu')(classification)
        # # classification = Dropout(0.5)(classification)
        # classification = Dense(len(self.__classes), activation='softmax')(classification)
        # return Model(inputs=base.input, outputs=classification)

# class ClassificationNetwork(NetworkArchitecture):
#     __VGG_NETWORKS = [VGG16, VGG19]
#     __RESNET_NETWORKS = [ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2]
#     __INCEPTION_NETWORKS = [InceptionV3]
#
#     def __init__(self, base, shape, classes):
#         self.base = base
#         self.__shape = shape
#         self.__classes = classes
#
#     @staticmethod
#     def standard_architecture(network, shape, classes):
#         if network in ClassificationNetwork.__VGG_NETWORKS:
#             return StandardVGGNetworkArchitecture(shape, classes, vgg=network)
#         elif network in ClassificationNetwork.__RESNET_NETWORKS:
#             return StandardResnetNetworkArchitecture(shape, classes, resnet=network)
#         elif network in ClassificationNetwork.__INCEPTION_NETWORKS:
#             return StandardInceptionNetworkArchitecture(shape, classes, inception=network)
#         else:
#             raise ValueError(f'Unknown base classification network: {network}')
#
#     def model(self):
#         base = self.base(weights='imagenet', include_top=False, input_shape=self.__shape)
#         base.trainable = False
#
#         if self.base == VGG16:
#             x = base.layers[-9].output
#             x = self.inception_module(x,
#                                       filters_1x1=192,
#                                       filters_3x3_reduce=96,
#                                       filters_3x3=208,
#                                       filters_5x5_reduce=16,
#                                       filters_5x5=48,
#                                       filters_pool_proj=64,
#                                       layer='inception-l1')
#             x = MaxPooling2D((2, 2), strides=(2, 2), name='inception_l1_maxpool')(x)
#
#             x = self.inception_module(x,
#                                       filters_1x1=160,
#                                       filters_3x3_reduce=112,
#                                       filters_3x3=224,
#                                       filters_5x5_reduce=24,
#                                       filters_5x5=64,
#                                       filters_pool_proj=64,
#                                       layer='inception-l2')
#             x = GlobalAveragePooling2D(name='inception_l2_avgpool')(x)
#
            # head = Flatten()(x)
            # classification = Dense(1024, activation='relu')(head)
            # classification = Dropout(0.35)(classification)
            # classification = Dense(512, activation='relu')(classification)
            # classification = Dropout(0.5)(classification)
            # classification = Dense(len(self.__classes), activation='softmax')(classification)
            # return Model(inputs=base.input, outputs=classification)
#
#         else:
#
#             head = Flatten()(base.output)
#             classification = Dense(512, activation='relu')(head)
#             classification = Dropout(0.5)(classification)
#             classification = Dense(512, activation='relu')(classification)
#             classification = Dropout(0.5)(classification)
#             classification = Dense(len(self.__classes), activation='softmax', name=CLASSIFICATION_NETWORK_NAME)(classification)
#             return Model(inputs=base.input, outputs=classification)
#
#     # Create the Inception module
#     def inception_module(self, x,
#                          filters_1x1,
#                          filters_3x3_reduce,
#                          filters_3x3,
#                          filters_5x5_reduce,
#                          filters_5x5,
#                          filters_pool_proj,
#                          layer,
#                          name=None):
#
#         # 1X1 CONV
#         conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
#                           bias_initializer=bias_init, name=f'{layer}-1-conv-1x1-1')(x)
#
#         # 1X1 CONV --> 3x3 CONV
#         conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
#                           bias_initializer=bias_init, name=f'{layer}-2-conv-1x1-1')(x)
#         conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init,
#                           bias_initializer=bias_init, name=f'{layer}-2-conv-3x3-2')(conv_3x3)
#
#         # 1X1 CONV --> 5x5 CONV
#         conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
#                           bias_initializer=bias_init, name=f'{layer}-3-conv-1x1-1')(x)
#         conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init,
#                           bias_initializer=bias_init, name=f'{layer}-3-conv-5x5-2')(conv_5x5)
#
#         # 3X3 MAXPOOL --> 1X1 CONV
#         pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name=f'{layer}-4-maxpool-1')(x)
#         pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init,
#                            bias_initializer=bias_init, name=f'{layer}-4-conv-1x1-2')(pool_proj)
#
#         # Concatenate the layers
#         output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
#
#         return output
#
#
# class ClassificationRegressionNetwork(NetworkArchitecture):
#     def __init__(self, base, shape, classes):
#         self.base = base
#         self.__shape = shape
#         self.__classes = classes
#
#     def model(self):
#         base = self.base(weights='imagenet', include_top=False, input_shape=self.__shape)
#         base.trainable = False
#
#         head = Flatten()(base.output)
#
#         # TODO: multiple detections per image
#         boundary = Dense(256, activation='relu')(head)
#         boundary = Dense(128, activation='relu')(boundary)
#         boundary = Dense(64, activation='relu')(boundary)
#         boundary = Dense(32, activation='relu')(boundary)
#         boundary = Dense(4, activation='sigmoid', name=BOUNDARY_NETWORK_NAME)(boundary)
#         # TODO: ensure input is scaled to 0,1 (x/255.0)
#
#         # TODO: multiple classifications per image
#         classification = Dense(512, activation='relu')(head)
#         classification = Dropout(0.5)(classification)
#         classification = Dense(512, activation='relu')(classification)
#         classification = Dropout(0.5)(classification)
#         classification = Dense(len(self.__classes), activation='softmax', name=CLASSIFICATION_NETWORK_NAME)(classification)
#         return Model(inputs=base.input, outputs=(boundary, classification))
