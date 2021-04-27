import argparse
import os
import keras
import tensorflow as tf
from typing import Optional
from constants import IMAGE_SIZE, IMAGE_CHANNELS, MASK_DETECTOR_ASHISH, MASK_DETECTOR_ANDREW, ALL_MASK_DETECTORS
from tools.job.train_job import train

__root = os.path.abspath('..')
ROOT_INPUT_LOCATION: str = os.path.join(__root, 'data')
ROOT_OUTPUT_LOCATION: str = os.path.join(__root, 'models', 'mask')


DEFAULT_EPOCHS: int = 15
DEFAULT_NETWORK: str = 'vgg16'
DEFAULT_DATASET: str = MASK_DETECTOR_ANDREW

__parser = argparse.ArgumentParser()
__parser.add_argument('--network', default=DEFAULT_NETWORK, type=str, help='The network architecture to use for bootstrapping')
__parser.add_argument('--modify_base', default=True, type=bool, help='Whether or not to use the custom or base network implementation')
__parser.add_argument('--epochs', default=DEFAULT_EPOCHS, type=int, help='The number of iterations')
__parser.add_argument('--dataset', default=DEFAULT_DATASET, type=str, help='Dataset to train model with')
__parser.add_argument('--checkpoint', type=str, help='The checkpoint to load from')
__parser.add_argument('--discriminator', type=str, help='The run discriminator of this training session')
__parser.add_argument('--resume', default=True, type=bool, help='Resume training from specified checkpoint')
# TODO: individual configs..?

args = __parser.parse_args()

print(f'Args: {args}')
print(f'Tensorflow API Version: {tf.__version__}')
print(f'Keras API Version: {keras.__version__}')
print(f'GPUS: {tf.config.list_physical_devices("GPU")}')

network: str = args.network
modify_base: bool = args.modify_base
checkpoint: Optional[str] = args.checkpoint
discriminator: Optional[str] = args.discriminator
dataset: str = args.dataset
epochs: int = args.epochs
resume: bool = args.resume

SHAPE = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
output_directory_base = os.path.join(ROOT_OUTPUT_LOCATION, dataset)

model, history, architecture, classes = train(
    root_input=ROOT_INPUT_LOCATION,
    root_output=ROOT_OUTPUT_LOCATION,
    shape=SHAPE,
    epochs=epochs,
    resume=resume,
    network=network,
    modify_base=modify_base,
    dataset=dataset,
    checkpoint=checkpoint,
    run_discriminator=discriminator
)
