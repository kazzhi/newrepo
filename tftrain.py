import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import collections
from tensorflow.keras import regularizers

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


val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),  # Load images at 224x224 first
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split = 0.2,
    subset = "validation",
    seed = 123
).map(preprocess_image)  # Apply preprocessing


# Load the training dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Save class names before mapping
train_class_names = train_dataset.class_names

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_image)

# Load the test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Save class names before mapping
test_class_names = test_dataset.class_names

# Apply preprocessing
test_dataset = test_dataset.map(preprocess_image)

# Print class names
print("Training dataset class names:", train_class_names)
print("Test dataset class names:", test_class_names)

#test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y))


def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """Optimized CNN model for ESP32-S3 with ESP-NN support."""
    model = keras.Sequential([
        # 1. First standard convolution
        layers.Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),  # Downsample

        # Second Block: Depthwise + Pointwise Conv
        layers.DepthwiseConv2D((3, 3), activation="relu", padding="same"),  
        layers.Conv2D(32, (1, 1), activation="relu", padding="same"),  
        #  layers.MaxPooling2D((2, 2)),  # Downsample

        # Third Block: Depthwise + Pointwise Conv
        layers.DepthwiseConv2D((3, 3), activation="relu", padding="same"),  
        layers.Conv2D(64, (1, 1), activation="relu", padding="same"),  
        layers.MaxPooling2D((2, 2)),  # Downsample

        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(256, activation="relu",  kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
         layers.Dropout(0.5), # Reduce overfitting
        #layers.BatchNormalization(),
        layers.Dense(num_classes, activation="softmax"),
       

        
        
    ])
    return model

model = build_cnn()
optimizer = keras.optimizers.Adam(learning_rate=0.0003, decay=1e-6)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
model.summary()

# Data augmentation for training
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    #horizontal_flip=True
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

print("Unique train labels:", np.unique(y_train))
print("Unique validation labels:", np.unique(y_val))
train_label_counts = collections.Counter(y_train)
print("Class Distribution:", train_label_counts)

# Check if training and test labels match
train_labels = np.unique(y_train)
test_labels = np.unique(np.concatenate([y.numpy() for _, y in test_dataset]))

print("Unique train labels:", train_labels)
print("Unique test labels:", test_labels)

# Verify the mapping
assert np.array_equal(train_labels, test_labels), "🚨 Label mismatch between training and test sets!"



class_counts = collections.Counter(y_train)
total = sum(class_counts.values())
class_weights = {cls: total/(count * NUM_CLASSES) for cls, count in class_counts.items()}

# # Include in model.fit()
# # Train the model
# early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[early_stop], class_weight=class_weights)

# # Evaluate on the test dataset
# test_loss, test_accuracy = model.evaluate(test_dataset)
# print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model.save("optimized_asl_model")