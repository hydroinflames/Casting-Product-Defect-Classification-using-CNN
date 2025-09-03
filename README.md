# Casting Product Defect Classification using CNN

![GitHub](https://img.shields.io/github/license/hydroinflames/Casting-Product-Defect-Classification-using-CNN)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Python](https://img.shields.io/badge/Python-3.7+-blue)

An end-to-end Convolutional Neural Network (CNN) for detecting manufacturing defects in industrial casting products with high accuracy.
(https://i.imgur.com/gU89aB1.png)


*Left: Defective sample (def_front) | Right: OK sample (ok_front)*

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [Future Improvements](#future-improvements)
- [License](#license)

## 🔍 Overview

This project implements a deep learning solution for automated quality control in manufacturing. Using a custom CNN architecture, the model can classify casting products as defective or non-defective based on their surface images. The solution achieves over 99% accuracy, making it viable for real-world industrial applications.

## ✨ Key Features

- **Data Augmentation Pipeline**: Enhances model generalization through on-the-fly image transformations using `ImageDataGenerator` (rotation, zoom, shift, flip)
- **Modern CNN Architecture**: Deep network with BatchNormalization, GELU activation functions, and Dropout for regularization
- **Robust Training Process**:
  - **AdamW Optimizer**: Improved regularization with weight decay
  - **Strategic Callbacks**: EarlyStopping to prevent overfitting, ReduceLROnPlateau for dynamic learning rate adjustment, and ModelCheckpoint to save the best model
- **Comprehensive Evaluation**: Performance assessed through accuracy/loss plots, confusion matrix, classification report, and ROC/AUC curve
- **Reproducibility**: Fixed random seeds in `random`, `numpy`, and `tensorflow` libraries for consistent results across different systems

## 💾 Dataset

The model is trained on the "Real-Life Industrial Dataset of Casting Product" from Kaggle, which contains images of real industrial casting components:

- Two classes: 'def_front' (defective) and 'ok_front' (non-defective)
- Grayscale images showing the surface of casting products
- Dataset Link: [Kaggle Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

## 🏗️ Model Architecture

The implemented CNN architecture follows a progressive structure:

```
Input → Conv2D → BatchNorm → GELU → MaxPooling → 
       → Conv2D → BatchNorm → GELU → MaxPooling → 
       → ... 
       → Flatten → Dense → BatchNorm → GELU → Dropout → 
       → Dense → BatchNorm → GELU → Dropout → 
       → Output (Dense, sigmoid)
```

Key components:
- Multiple convolutional blocks with increasing filter counts
- Batch normalization after each convolutional and dense layer
- GELU activation functions for improved gradient flow
- Dropout layers for regularization in the fully connected sections
- Binary classification output with sigmoid activation

## 📊 Results

The model achieves exceptional performance on the test dataset:

- **Accuracy**: >99%
- **Precision & Recall**: High values for both defective and non-defective classes
- **F1-Score**: Nearly perfect balance between precision and recall

*Note: Add your training curve, confusion matrix, and ROC curve images here after generating them.*

<!-- Example placeholder for visualization images:
**Training and Validation Curves**
![Training Curves](path_to_your_training_plot.png)

**Confusion Matrix**
![Confusion Matrix](path_to_your_confusion_matrix.png)

**ROC Curve**
![ROC Curve](path_to_your_roc_curve.png)
-->


## 💡 Future Improvements

Potential enhancements for this project:

- **Transfer Learning**: Leverage pre-trained models like ResNet50 or EfficientNet for potentially higher accuracy
- **Advanced Data Augmentation**: Experiment with more sophisticated techniques like CutMix and Mixup
- **Hyperparameter Optimization**: Use KerasTuner or Optuna for automated hyperparameter tuning
- **Model Deployment**: Serve the model using TensorFlow Serving, Flask, or FastAPI for real-time inference
- **Explainable AI**: Implement GradCAM or similar techniques to visualize which parts of the images contribute most to the classification decision

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
