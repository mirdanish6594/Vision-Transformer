# Vision Transformer

## Overview
This repository contains an implementation of the Vision Transformer (ViT) architecture for image classification using the CIFAR-10 dataset. The provided code is intended to showcase how transformers can be applied to visual tasks, leveraging self-attention mechanisms to capture both local and global features in images.

## Code Description
The main code is implemented in a Google Colab notebook, which includes the following key components:

1. **Data Loading and Preprocessing:** The CIFAR-10 dataset is loaded, resized to 72x72 pixels, normalized, and augmented using techniques such as random flipping, rotation, and zooming.
  
2. **Vision Transformer Model:** The model architecture is defined, which includes:
   - Splitting images into patches.
   - Encoding patches with a projection layer and positional embeddings.
   - Applying multiple transformer layers that utilize multi-head self-attention mechanisms.
   - Using a multi-layer perceptron (MLP) head for classification.

3. **Training and Evaluation:** The model is trained using the AdamW optimizer with weight decay. After training, the model's performance is evaluated on the test set, providing metrics such as accuracy and top-5 accuracy.

## Usage
You can explore the implementation directly in the provided Google Colab notebook. To run the notebook:
1. Open the notebook in Google Colab by clicking on the link in this repository.
2. Follow the instructions in the notebook to execute the code, train the model, and evaluate its performance.

## Requirements
To run the notebook locally, ensure you have the following packages installed:
- TensorFlow 2.8.0
- Keras 2.8.0
- TensorFlow Addons 0.17.0
- NumPy
- Matplotlib

You can install the required packages using pip:
```bash
!pip install tensorflow==2.8.0
!pip install keras==2.8.0
!pip install tensorflow-addons==0.17.0 #(0.20.0)
```
## Results
The model demonstrates competitive performance on the CIFAR-10 dataset, showcasing the capability of the Vision Transformer architecture for image classification tasks.






