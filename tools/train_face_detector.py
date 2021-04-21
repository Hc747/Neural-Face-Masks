import argparse
import os
import tensorflow as tf
from constants import IMAGE_SIZE, IMAGE_CHANNELS, MASK_DETECTOR_ASHISH, MASK_DETECTOR_ANDREW


__root = os.path.abspath('..')
ROOT_INPUT_LOCATION: str = os.path.join(__root, 'data')
ROOT_OUTPUT_LOCATION: str = os.path.join(__root, 'models', 'mask')


DEFAULT_EPOCHS: int = 15
DEFAULT_NETWORK: str = 'vgg16'
DEFAULT_DATASET: str = 'andrew'
DEFAULT_DUPLICATE: bool = False
DEFAULT_REGRESSION: bool = False

__parser = argparse.ArgumentParser()
__parser.add_argument('--network', default=DEFAULT_NETWORK, type=str, help='The network architecture to use for bootstrapping')
__parser.add_argument('--epochs', default=DEFAULT_EPOCHS, type=int, help='The number of iterations')
__parser.add_argument('--dataset', default=DEFAULT_DATASET, type=str, help='Dataset to train model with')
__parser.add_argument('--duplicate', default=DEFAULT_DUPLICATE, type=bool, help='Use duplicate images')
__parser.add_argument('--regression', default=DEFAULT_REGRESSION, type=bool, help='Train a classification regression network instead of a classification network')
# TODO: individual configs..?

args = __parser.parse_args()

print(f'Args: {args}')

dataset = args.dataset
SHAPE = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
output_directory_base = os.path.join(ROOT_OUTPUT_LOCATION, dataset)

if dataset == MASK_DETECTOR_ANDREW:
    from tools.dataset.andrew import generate
elif dataset == MASK_DETECTOR_ASHISH:
    from tools.dataset.ashish import generate
else:
    raise ValueError(f'Unknown dataset: {dataset}')

(x, y, validation), (architecture, model, classes, output) = generate(
    SHAPE,
    ROOT_INPUT_LOCATION,
    output_directory_base,
    args
)

model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=output,
    monitor='val_loss',
    mode='auto',
    save_weights_only=False,
    save_best_only=True,
    verbose=1
)

history = model.fit(
    x,
    y,
    validation_data=validation,
    epochs=args.epochs,
    callbacks=[checkpoint],
    verbose=1
)
