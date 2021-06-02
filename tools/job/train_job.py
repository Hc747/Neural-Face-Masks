import os
import time
import tensorflow as tf
from typing import Optional, Tuple
from tensorflow.keras.applications import *
from constants import *
from network.network_architecture import ClassificationNetwork, CLASSIFICATION_NETWORK_NAME, LOSS_FUNCTIONS

"""
A module providing functionality for training models in a consistent manner.
"""

CHECKPOINT_DELIMITER: str = ':'


def train(root_input: str, root_output: str, shape: Tuple[int, int, int], epochs: int, resume: bool, description: str, dataset: str, checkpoint: Optional[str] = None, run_discriminator: Optional[str] = None):
    """
    Encapsulates the functionality for building and training a model.

    :param root_input:
    The base input directory (so as to allow this code to be environment agnostic).
    :param root_output:
    The base output directory (so as to allow this code to be environment agnostic).
    :param shape:
    The shape of the images in the dataset.
    :param epochs:
    The number of epochs to train for.
    :param resume:
    Whether or not to continue training from where we left off.
    :param description:
    A descriptor for the model we are building.
    :param dataset:
    The dataset to use for training. Can be one of MASK_DETECTOR_ASHISH or MASK_DETECTOR_CABANI.
    :param checkpoint:
    A ':' delimited string containing the descriptor and epoch of the checkpoint to resume from.
    :param run_discriminator:
    A string that uniquely identifies this run - supplied if we need it to be human readable.

    :return:
    The model, training history, network architecture, class mapping and training data (if any).
    """

    if dataset == MASK_DETECTOR_ASHISH:
        from tools.dataset.ashish import generate
    elif dataset == MASK_DETECTOR_CABANI:
        from tools.dataset.cabani import generate
    else:
        raise ValueError(f'Unknown dataset: {dataset}. Must be one of: {ALL_MASK_DETECTORS}')

    dataset_output: str = os.path.join(root_output, dataset)
    discriminator: str = f'{time.time_ns() // 1_000_000}' if run_discriminator is None else run_discriminator
    extension: str = f'{description}' + '-checkpoint-{epoch:04d}'

    ((x_train, y_train), (x_validation, y_validation), (x_test, y_test)), (classes, output) = generate(
        shape=shape,
        _input=root_input,
        _output=dataset_output
    )

    print('-' * 5)
    print(f'classes: {classes}')
    print('-' * 5)

    # TODO: save class mapping
    architecture = ClassificationNetwork(base=VGG16, shape=IMAGE_SHAPE, classes=classes, trainable=False)
    model = architecture.compile(LOSS_FUNCTIONS[CLASSIFICATION_NETWORK_NAME], None)

    if checkpoint is not None:
        if checkpoint.count(CHECKPOINT_DELIMITER) != 1:
            raise ValueError(f'Expected checkpoint to contain 1 delimiter character: \'{CHECKPOINT_DELIMITER}\'')
        timestamp, epoch = checkpoint.split(':', 2)
        epoch = int(epoch)
        if resume:
            epochs -= epoch
        location = os.path.join(output, timestamp, extension).format(epoch=epoch)
        print(f'Updating weights from snapshot: {location}')
        snapshot = tf.keras.models.load_model(location)
        model.set_weights(snapshot.get_weights())
        print(f'Updated weights from snapshot: {location}')

    model.summary()

    persistence = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(output, discriminator, extension),
        monitor='val_loss',
        mode='auto',
        save_weights_only=False,
        save_best_only=True,
        verbose=1
    )

    if epochs >= 1:
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_validation, y_validation),
            epochs=epochs,
            callbacks=[persistence],
            verbose=1
        )
    else:
        history = None

    return model, history, architecture, classes, (x_test, y_test)
