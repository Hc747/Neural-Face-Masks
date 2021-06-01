# 42028
# Deep Learning and Convolutional Neural Networks
# Assignment 3
### Group 19
### Face the Facts

## Contributors
| Name | Email | Github | ID |
| ---- | ----- | ------ | ---------- |
| Harrison Cole | [email](mailto:harrisoncole05@gmail.com) | [Hc747](https://github.com/Hc747) | 12962712 |
| Jihee Lee | [email](mailto:ruaglzzz@gmail.com) | [GRORY](https://github.com/GRORY) | 13826920 |
| Jose Meza | [email](mailto:josemezag@gmail.com) | [jose-meza-garcia](https://github.com/jose-meza-garcia) | 13093099 |

## About
The wearing of face masks in public spaces has become a normal, if not mandatory, aspect of daily life in most countries around the world.
The proper usage of face masks has been demonstrated to be an effective measure in controlling the spread of the coronavirus disease (COVID-19).

Our project delivers a tool that is capable of locating faces within images and videos, detecting if face masks are present, and detecting if they are being worn correctly.

This application is an example of a tool that could be deployed in public spaces (i.e., shopping centres, airports, public transport) in order to provide interested parties with analytical and/or monitoring capabilities.
This tool _does not_ provide interested parties with the ability to correlate faces with identities.

### Dataset
Images were sourced from the following datasets.

| [Cabani](https://github.com/cabani/MaskedFace-Net) | [Ashish](https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset) | [Flickr](https://github.com/NVlabs/ffhq-dataset) |
| -------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------ |

| classes | masked | unmasked | uncovered_nose_and_mouth | uncovered_nose | uncovered_chin |
| ------- | :----: | :------: | :----------------------: | :------------: | :------------: |
| **samples** | 9,688 | 9,688 | 4,834 | 4,834 | 4,834 |

| ![](./docs/examples/1.jpg) | ![](./docs/examples/2.jpg) | ![](./docs/examples/3.jpg) |
| ---------------------------| ---------------------------| ---------------------------|
| ![](./docs/examples/4.jpg) | ![](./docs/examples/5.jpg) | ![](./docs/examples/6.jpg) |
| ![](./docs/examples/7.jpg) | ![](./docs/examples/8.jpg) | ![](./docs/examples/9.jpg) |

### Libraries
| [TensorFlow](https://github.com/tensorflow/tensorflow) | [MediaPipe](https://github.com/google/mediapipe) | [Dlib](https://github.com/davisking/dlib) | [OpenCV](https://github.com/opencv/opencv) | [scikit-image](https://scikit-image.org/) | [Pillow](https://python-pillow.org/) | [NumPy](https://numpy.org/) | [Tkinter](https://docs.python.org/3/library/tkinter.html) |
| ------------------------------------------------------ | ------------------------------------------------ | ----------------------------------------- | ------------------------------------------ | ----------------------------------------- | ------------------------------------ | --------------------------- | --------------------------------------------------------- |

### Architecture
Two models were developed using the following modified VGG16 CNN architecture.

![CNN Pipeline: Modified VGG16](docs/architecture/architecture.png)

## Getting Started
### Install
```python
pip install -r requirements.txt
```

### Run
```shell
python main.py
```

### Screenshots
![Masked](docs/app/masked-complex.jpg)
![Unmasked](docs/app/unmasked.jpg)
![Incorrectly Masked - Uncovered Nose](docs/app/masked-uncovered-nose.jpg)
![Incorrectly Masked - Uncovered Nose and Mouth](docs/app/masked-uncovered-nose-and-mouth.jpg)
![Incorrectly Masked - Uncovered Chin](docs/app/masked-uncovered-chin.jpg)
![Unprocessed](docs/app/masked-raw.jpg)
