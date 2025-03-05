import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
#import os
#from sklearn.model_selection import train_test_split

# Define constants
DATA_DIR = "Images/"
TEST_DIR = "Test/"
IMG_SIZE = 96  # Resized image size
BATCH_SIZE = 32
NUM_CLASSES = 28 # 26 letters + space + nothing

def preprocess_image(image, label):
    """Preprocess images: Convert to grayscale, resize, and normalize."""
    image = tf.image.rgb_to_grayscale(image)
    # image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

# Load dataset from directory
full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),  # Load images at 224x224 first
    batch_size=BATCH_SIZE,
    shuffle=True
).map(preprocess_image)  # Apply preprocessing


total_batches = len(full_dataset)
train_batches = int(total_batches * 0.8)
train_dataset = full_dataset.take(train_batches)
val_dataset = full_dataset.skip(train_batches)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),  # No resizing needed
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    shuffle=False
).map(preprocess_image)

def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """Optimized CNN model for ESP32-S3 with ESP-NN support."""
    model = keras.Sequential([
        # 1. First standard convolution
        layers.Conv2D(8, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),  # Reduce spatial size

        # 2. Second standard convolution (instead of separable conv)
        layers.Conv2D(16, (1, 1), activation="relu", padding="same"),
        # layers.MaxPooling2D((2, 2)),

        # 3. Third standard convolution
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (1, 1), activation="relu", padding="same"),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # 4. Fully Connected Layer
        layers.Flatten(),
        layers.Dense(64, activation="relu"),

        # 5. Output layer
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

model = build_optimized_cnn()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Data augmentation for training
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Convert datasets to NumPy for augmentation
X_train, y_train = [], []
X_val, y_val = [], []

for images, labels in train_dataset:
    X_train.append(images.numpy())
    y_train.append(labels.numpy())

for images, labels in val_dataset:
    X_val.append(images.numpy())
    y_val.append(labels.numpy())

X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
X_val = np.concatenate(X_val, axis=0)
y_val = np.concatenate(y_val, axis=0)

# Ensure correct shape for grayscale images
X_train = X_train.reshape(X_train.shape[0], IMG_SIZE, IMG_SIZE, 1)
X_val = X_val.reshape(X_val.shape[0], IMG_SIZE, IMG_SIZE, 1)

# Apply data augmentation
datagen.fit(X_train)
train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Evaluate on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model.save("optimized_asl_model")