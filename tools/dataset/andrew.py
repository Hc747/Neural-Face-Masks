import os
import xml.etree.ElementTree as et
import numpy as np
from typing import Tuple
from tensorflow.keras.applications import VGG16
from keras.utils.np_utils import to_categorical
from keras_preprocessing.image import img_to_array, load_img, ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from constants import IMAGE_SIZE, RANDOM_STATE
from network.network_architecture import ClassificationRegressionNetwork, LOSS_FUNCTIONS, LOSS_WEIGHTS, \
    ClassificationNetwork, CLASSIFICATION_NETWORK_NAME, BOUNDARY_NETWORK_NAME, NETWORKS
from tools.dataset.common import augmentor

DUPLICATE: bool = False
REGRESSION: bool = False


def generate(shape: Tuple[int, int, int], network: str, modify_base: bool, _input: str, _output: str):
    base_directory: str = os.path.join(_input, 'kaggle', 'andrewmvd')

    image_directory: str = os.path.join(base_directory, 'images')
    annotation_directory: str = os.path.join(base_directory, 'annotations')

    images = []
    labels = []
    boundaries = []

    for index, file in enumerate(os.listdir(annotation_directory)):
        annotation = et.parse(os.path.join(annotation_directory, file)).getroot()

        filename: str = annotation.find('filename').text
        path: str = os.path.join(image_directory, filename)

        width: int = int(annotation.find('size/width').text)
        height: int = int(annotation.find('size/height').text)

        image = img_to_array(load_img(path, target_size=shape)) / 255.0

        scaled_width: float = float(IMAGE_SIZE / width)
        scaled_height: float = float(IMAGE_SIZE / height)

        for boundary in annotation.iter('object'):
            label: str = boundary.find('name').text

            xmin = float(boundary.find('bndbox/xmin').text)
            ymin = float(boundary.find('bndbox/ymin').text)
            xmax = float(boundary.find('bndbox/xmax').text)
            ymax = float(boundary.find('bndbox/ymax').text)

            # scaling, normalisation, sigmoid & image duplication
            # ssd instead of faster-r-cnn

            # point = (original * scale) / new
            scaled_xmin = (xmin * scaled_width) / IMAGE_SIZE
            scaled_ymin = (ymin * scaled_height) / IMAGE_SIZE
            scaled_xmax = (xmax * scaled_width) / IMAGE_SIZE
            scaled_ymax = (ymax * scaled_height) / IMAGE_SIZE

            coordinates = (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax)

            images.append(image)
            labels.append(label)
            boundaries.append(coordinates)

            if not DUPLICATE:
                break

    images = np.asarray(images)
    labels = np.asarray(labels)
    boundaries = np.asarray(boundaries)
    classes, totals = np.unique(labels, return_counts=True)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    labels = to_categorical(labels)

    splits = train_test_split(images, labels, boundaries, test_size=0.2, random_state=RANDOM_STATE)

    (train_images, test_images) = splits[0:2]
    (train_labels, test_labels) = splits[2:4]
    (train_boundaries, test_boundaries) = splits[4:6]

    base = NETWORKS.get(network, VGG16)

    if REGRESSION:
        output = os.path.join(_output, 'regression', 'checkpoint')
        train_targets = {
            BOUNDARY_NETWORK_NAME: train_boundaries,
            CLASSIFICATION_NETWORK_NAME: train_labels
        }

        test_targets = {
            BOUNDARY_NETWORK_NAME: test_boundaries,
            CLASSIFICATION_NETWORK_NAME: test_labels
        }

        x = train_images
        y = train_targets
        training = (x, y)
        validation = (test_images, test_targets)

        architecture = ClassificationRegressionNetwork(base=base, shape=shape, classes=classes)
        model = architecture.compile(loss=LOSS_FUNCTIONS, weights=LOSS_WEIGHTS)

    else:
        output = os.path.join(_output, 'classification', 'checkpoint')
        generator = augmentor
        x = generator.flow(train_images, train_labels, seed=RANDOM_STATE)
        y = None
        training = (x, y)
        validation = (test_images, test_labels)

        if modify_base:
            architecture = ClassificationNetwork(base=base, shape=shape, classes=classes)
        else:
            architecture = ClassificationNetwork.standard_architecture(network=base, shape=shape, classes=classes)

        model = architecture.compile(LOSS_FUNCTIONS[CLASSIFICATION_NETWORK_NAME], None)

    return (training, validation), (architecture, model, classes, output), (images, labels, boundaries)
