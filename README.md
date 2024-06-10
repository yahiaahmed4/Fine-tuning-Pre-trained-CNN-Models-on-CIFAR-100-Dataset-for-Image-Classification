# Fine-tuning Pre-trained CNN Models on CIFAR-100 Dataset for Image Classification
 
This repository contains three notebooks that demonstrate how to fine-tune pre-trained Convolutional Neural Network (CNN) models available in Keras for the task of image classification using the CIFAR-100 dataset.

## Notebooks Overview:

### Notebook 1: Fine-tuning VGG16 on CIFAR-100
- **Model**: VGG16
- **Dataset**: CIFAR-100
- **Preprocessing**: Images are normalized and categorical labels are one-hot encoded.
- **Training**: The model is trained using transfer learning by loading the pre-trained VGG16 model without the top (fully connected) layers. The top layers are replaced with custom fully connected layers for CIFAR-100 classification.
- **Results**: Achieves a test accuracy of approximately 77.51%.

### Notebook 2: Fine-tuning DenseNet201 on CIFAR-100
- **Model**: DenseNet201
- **Dataset**: CIFAR-100
- **Preprocessing**: Images are normalized and categorical labels are one-hot encoded.
- **Training**: Similar to Notebook 1, the DenseNet201 model is loaded without the top layers and replaced with custom layers for CIFAR-100 classification.
- **Results**: Achieves a test accuracy of around 78.32%.

### Notebook 3: Fine-tuning EfficientNetV2L on CIFAR-100
- **Model**: EfficientNetV2L
- **Dataset**: CIFAR-100
- **Preprocessing**: Images are normalized and categorical labels are one-hot encoded.
- **Training**: The EfficientNetV2L model is loaded without the top layers and replaced with custom layers for CIFAR-100 classification.
- **Results**: Attains a test accuracy of approximately 78.48%.

## Dataset:
The CIFAR-100 dataset consists of 60,000 32x32 color images in 100 classes, with 600 images per class. The dataset is divided into 50,000 training images and 10,000 test images.

## Preprocessing:
- Images are normalized by dividing pixel values by 255.0 to scale them between 0 and 1.
- Categorical labels are one-hot encoded to represent the target classes.

## Training:
- Training is performed using transfer learning, where pre-trained CNN models are adapted to the CIFAR-100 dataset by replacing their top layers.
- The models are compiled with appropriate loss functions, optimizers, and evaluation metrics.
- Training is monitored using early stopping to prevent overfitting, and learning rate reduction callbacks are employed to adjust the learning rate if validation accuracy plateaus.

## Results:
- The notebooks report the final test accuracy achieved by each model on the CIFAR-100 test set.
- Additionally, sample predictions are generated for visualization purposes.

For more details, refer to the individual notebooks in this repository. Enjoy exploring and experimenting with image classification using pre-trained CNN models!
