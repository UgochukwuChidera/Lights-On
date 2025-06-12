# Transfer Learning Example in PyTorch and TensorFlow
# ---------------------------------------------------
# This script demonstrates basic transfer learning using two popular frameworks:
# 1. PyTorch (with ResNet50)
# 2. TensorFlow (with MobileNetV2)
#
# For both, we:
# - Load a pretrained model
# - Replace the final classification layer to match a new number of classes
# - Freeze the base layers (optional)
# - Retrain the model on new data (dummy example for illustration)

# ----------------------
# PyTorch Implementation
# ----------------------
import torch
import torch.nn as nn
from torchvision import models

# Number of classes for the new task (e.g., 10 for CIFAR-10)
NUM_CLASSES = 10

# Load a pretrained ResNet50 model
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze all layers (optional, for feature extraction)
for param in resnet.parameters():
    param.requires_grad = False

# Replace the final fully connected layer to match NUM_CLASSES
# The original layer is: resnet.fc = nn.Linear(2048, 1000)
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)

# Only the new layer's parameters will be updated during training
optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=0.001)

# Example dummy input and target for illustration
x = torch.randn(4, 3, 224, 224)  # batch of 4 images
labels = torch.randint(0, NUM_CLASSES, (4,))

# Forward pass
outputs = resnet(x)

# Compute loss
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, labels)

# Backward pass and optimization
loss.backward()
optimizer.step()

# --------------------------
# TensorFlow Implementation
# --------------------------
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Number of classes for the new task
NUM_CLASSES = 10

# Load a pretrained MobileNetV2 model, excluding the top (classification) layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Add new classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Example dummy data for illustration
import numpy as np
x = np.random.randn(4, 224, 224, 3).astype(np.float32)
labels = np.random.randint(0, NUM_CLASSES, 4)

# Train on dummy data (1 step)
model.train_on_batch(x, labels)

# --------------------------
# Notes:
# - In practice, replace dummy data with your real dataset and use DataLoader (PyTorch) or tf.data (TensorFlow).
# - You can unfreeze some base layers for fine-tuning by setting requires_grad=True (PyTorch) or base_model.trainable=True (TensorFlow).
# - Always preprocess your data as required by the pretrained model.
