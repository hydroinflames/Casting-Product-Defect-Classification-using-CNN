# -*- coding: utf-8 -*-
"""
Casting Defect Classification — A Robust & Shareable Script

This script builds, trains, and evaluates a CNN model for classifying casting defects.
Key features include:
- Clear, sectioned workflow from data loading to evaluation.
- Reproducibility with set random seeds.
- On-the-fly data augmentation to prevent overfitting.
- AdamW optimizer and modern activation functions (GELU).
- Robust training loop with callbacks like EarlyStopping and ModelCheckpoint.
- Comprehensive evaluation including a confusion matrix, classification report, and ROC/AUC curve.
- Ready for professional sharing on platforms like GitHub and LinkedIn.
"""

# ========================= 0. Imports & Seeds =========================
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW

# Install opendatasets for easy Kaggle data download
!pip install opendatasets --upgrade --quiet
import opendatasets as od

# For reproducibility (note: some GPU operations can still be non-deterministic)
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ========================= 1. Data Loading & Initial Exploration =========================

# Download the dataset from Kaggle
dataset_url = 'https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product'
print(f"Downloading dataset from: '{dataset_url}'...")
od.download(dataset_url, force=True) # Use force=True to overwrite if it exists
print("Dataset downloaded successfully.")

# Define paths
base_dir = 'real-life-industrial-dataset-of-casting-product/casting_data/casting_data'
train_dir = os.path.join(base_dir, 'train')
test_dir  = os.path.join(base_dir, 'test')

def_front_dir = os.path.join(train_dir, 'def_front')
ok_front_dir  = os.path.join(train_dir, 'ok_front')

# Sanity check to ensure directories exist
assert os.path.isdir(def_front_dir) and os.path.isdir(ok_front_dir), "Directory structure error. 'def_front' or 'ok_front' not found."

# Preview a random pair of images (defective vs. ok)
rand_def = os.path.join(def_front_dir, random.choice(os.listdir(def_front_dir)))
rand_ok  = os.path.join(ok_front_dir,  random.choice(os.listdir(ok_front_dir)))

def imshow_pair(path1, path2):
    """Displays a pair of images for comparison."""
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(load_img(path1))
    ax1.set_title('Defective (def_front)', fontsize=14)
    ax1.axis('off')
    ax2.imshow(load_img(path2))
    ax2.set_title('OK (ok_front)', fontsize=14)
    ax2.axis('off')
    fig.suptitle('Sample Image Comparison', fontsize=18)
    plt.show()

imshow_pair(rand_def, rand_ok)


# ========================= 2. Data Preprocessing & Augmentation =========================
IMG_HEIGHT = 150
IMG_WIDTH  = 150
BATCH_SIZE = 32

# Define an ImageDataGenerator for training with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2,  # Split 20% of training data for validation
)

# Test generator should only rescale the images, no augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    subset='training',  # Specify this is the training subset
    seed=SEED,
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False,  # No need to shuffle validation data
    subset='validation', # Specify this is the validation subset
    seed=SEED,
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=1,  # Process one image at a time for evaluation
    class_mode='binary',
    shuffle=False,
)

print("\nClass Indices:", train_generator.class_indices)


# ========================= 3. CNN Model Architecture =========================
model = Sequential([
    # Block 1
    Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Block 2
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Block 3
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Block 4
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # Flatten the feature map
    Flatten(),

    # Dense Head with GELU activation and Dropout for regularization
    Dense(512, activation=tf.nn.gelu),
    BatchNormalization(),
    Dropout(0.5),

    Dense(256, activation=tf.nn.gelu),
    BatchNormalization(),
    Dropout(0.5),

    # Output layer
    Dense(1, activation='sigmoid'),
])

# Compile the model with AdamW optimizer
model.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ========================= 4. Model Training =========================
EPOCHS = 25 # Increased epochs slightly as EarlyStopping will handle it

# Define callbacks for robust training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_casting_cnn.keras', monitor='val_loss', save_best_only=True)
]

print("\nStarting model training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1,
)
print("Model training finished.")


# ========================= 5. Plotting Training Curves =========================
history_dict = history.history
acc = history_dict.get('accuracy', [])
val_acc = history_dict.get('val_accuracy', [])
loss = history_dict.get('loss', [])
val_loss = history_dict.get('val_loss', [])

# Plot Accuracy and Loss
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy', color='blue')
plt.plot(val_acc, label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='orange')
plt.title('Training and Validation Loss', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.suptitle('Model Performance Curves', fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# ========================= 6. Final Evaluation on Test Data =========================
print("\n--- Evaluating on Test Data ---")
results = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")

# Get predictions and true labels
probs = model.predict(test_generator, verbose=1).ravel()  # Flatten to shape (N,)
predictions = (probs > 0.5).astype(int)
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.show()

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(true_labels, predictions, target_names=class_labels))

# ROC Curve and AUC Score
fpr, tpr, thresholds = roc_curve(true_labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"AUC Score: {roc_auc:.4f}")

# ========================= 7. Further Improvement Tips =========================
print("\n--- Tips for Further Improvement ---")
print("- Experiment with different CNN architectures (e.g., ResNet, EfficientNet via transfer learning).")
print("- Fine-tune hyperparameters like learning rate, batch size, and optimizer settings.")
print("- If class imbalance exists, consider using class weights during training.")
print("- Explore more advanced data augmentation techniques (e.g., CutMix, Mixup).")
print("- For very large datasets, optimize the input pipeline with tf.data and prefetching.")