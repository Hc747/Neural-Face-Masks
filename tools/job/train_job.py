import os
import time
import tensorflow as tf
from typing import Optional, Tuple
from keras.callbacks import Callback
from constants import *

CHECKPOINT_DELIMITER: str = ':'


def train(root_input: str, root_output: str, shape: Tuple[int, int, int], epochs: int, resume: bool, network: str, modify_base: bool, dataset: str, checkpoint: Optional[str] = None, run_discriminator: Optional[str] = None):
    if dataset == MASK_DETECTOR_ANDREW:
        from tools.dataset.andrew import generate
    elif dataset == MASK_DETECTOR_ASHISH:
        from tools.dataset.ashish import generate
    else:
        raise ValueError(f'Unknown dataset: {dataset}. Must be one of: {ALL_MASK_DETECTORS}')

    dataset_output: str = os.path.join(root_output, dataset)
    discriminator: str = f'{time.time_ns() // 1_000_000}' if run_discriminator is None else run_discriminator
    extension: str = f'{network}' + '-checkpoint-{epoch:04d}'

    ((x_train, y_train), (x_validation, y_validation), (x_test, y_test)), (architecture, model, classes, output) = generate(
        shape=shape,
        network=network,
        modify_base=modify_base,
        _input=root_input,
        _output=dataset_output
    )

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

    persistence: Callback = tf.keras.callbacks.ModelCheckpoint(
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
