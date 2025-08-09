# A 3D Upper Airway Atlas: Data, Segmentations, and Printable Vocal Tracts

## Introduction
This repository contains the implementation of a 3D U-Net architecture for segmenting upper airway structures from 3D medical images. The model is designed to assist in creating a comprehensive dataset of segmented upper airway images, which can be further refined with expert input.

## Dataset Acquisition

- The dataset used in this project consists of 3D NRRD images of upper airway structures.
- The training and validation images are stored in the following directories:
  - Training Data: `./Data/VoiceUsers/Train/Train`
  - Validation Data: `./Data/VoiceUsers/Val/Nasal25`

## Model Architecture
- A 3D U-Net model is utilized to perform the segmentation task. The model architecture consists of multiple layers composed of convolutional operations, batch normalization, and ReLU activations.

## Installation
Ensure you have the following packages installed:
- `torch`
- `monai`
- `numpy`
- `matplotlib`
- `nrrd`

You can install the required packages using pip:
```bash
pip install torch monai numpy matplotlib nrrd
```

## Training
To train the model, use the provided Python script `VanillaUNet3D.py`. Adjust the parameters, such as learning rate and number of epochs, to suit your needs. The current script is configured to run for 1500 epochs.

### Training Steps:
1. Load the training and validation datasets.
2. Apply transformations to the images and labels, including intensity normalization and random cropping.
3. Define the loss function and optimizer.
4. Train the model and periodically validate its performance on the validation dataset.

## Evaluation
- After training, the model's performance can be evaluated using the Dice metric, which measures the overlap between the predicted segmentation and the ground truth.
- The best-performing model is saved for future use.

## Contribution
Experts are encouraged to provide feedback and refine the segmentation results produced by the model. This collaborative effort will help improve the quality of the generated segmentations.

## License
This project is licensed under the MIT License.