# Totally Looks Like: Siamese Network with ResNet50 and VGG19

The "Totally Looks Like" project aims to identify visually similar images using a Siamese neural network. The network combines features extracted from two pre-trained models, ResNet50 and VGG19, to enhance its similarity matching capabilities.


## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset;)
- [Installation](#installation)
- [Usage](#usage)
- [Advance Configuration](#advance-configuration)
- [Features](#features)


## Dataset
-   In this project we have utilized the Totally looks like dataset by Amir Rosenfeld, Markus D. Solbach, John K. Tsotsos
-   Follow the link to download the dataset for train, evaluate, and test.
-   https://sites.google.com/view/totally-looks-like-dataset
## Installation
Install the required packages:

```bash
pip install -r requirements.txt
```

Please note 
-   If you want to use  GPU for this notebook with tensorflow, please follow the steps listed on the official website and install all required packages https://www.tensorflow.org/install/pip
-   Otherwise by just install requirements would be sufficient to run with cpu only. 

## Usage
All work is done in a Jupyter Notebook. To begin, launch Jupyter Notebook in the project directory:
```bash
jupyter notebook
```
Navigate to the notebook named CV.ipynb and execute the cells to train and evaluate the mode

## Advance Configuration
You can customize various parameters directly in the Jupyter Notebook, such as:

-   Number of training epochs.
-   Size of each mini-batch.
-   Learning rate for the optimizer.

## Features
- **Dual Model Architecture**: Uses ResNet50 for capturing higher-level features and VGG19 for more nuanced, texture-based features.
- **Custom Loss Function**: Employs a custom contrastive loss function to optimize the learning process.
- **High Flexibility**: Supports various configurations like the number of epochs, batch size, and learning rates.
- **Evaluation Metrics**: Inclusion of evaluation scripts to assess model performance based on Top-K Categorical Accuracy.
- **TensorFlow TPU Support**: Optimized for TensorFlow and capable of running on TPUs for faster computation.