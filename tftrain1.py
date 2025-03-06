import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import collections
import os
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt



#import os
#from sklearn.model_selection import train_test_split

# Define constants
DATA_DIR = "Images/"
TEST_DIR = "Test/"
IMG_SIZE = 96  # Resized image size
BATCH_SIZE = 16
NUM_CLASSES = 27 # 26 letters + space + nothing

def preprocess_image(image, label):
    """Preprocess images: Convert to grayscale, resize, and normalize."""
    image = tf.image.rgb_to_grayscale(image)
    # image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

# Load dataset from directory
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),  # Load images at 224x224 first
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split = 0.8,
    subset = "training",
    seed = 123
).map(preprocess_image)  # Apply preprocessing



val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),  # Load images at 224x224 first
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split = 0.2,
    subset = "validation",
    seed = 123
).map(preprocess_image)  # Apply preprocessing


test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),  # No resizing needed
    batch_size=BATCH_SIZE,
    shuffle=False
).map(preprocess_image)

print("Training folder names:", os.listdir(DATA_DIR))
print("Test folder names:", os.listdir(TEST_DIR))

#test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y))


def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """Optimized CNN model for ESP32-S3 with ESP-NN support."""
    model = keras.Sequential([
        layers.Conv2D(8, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(16, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        #layers.Dropout(0.4),
        
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        #layers.Dropout(0.4),

        #layers.Flatten(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.005)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(27, activation='softmax', kernel_regularizer=l2(0.002))  # 28 classes
       
        
    ])
    return model

model = build_cnn()
optimizer = keras.optimizers.Adam(learning_rate=3e-4, decay=1e-6)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
model.summary()

# Data augmentation for training
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    #brightness_range=[0.9, 1.1]
    #horizontal_flip=True
    #contrast_stretching=True
    zoom_range = 0.1
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
#datagen.fit(X_train)
train_generator = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

# for images, labels in test_dataset.take(3):
#     print("Labels:", labels.numpy())

class_counts = collections.Counter(y_train)
total = sum(class_counts.values())
class_weights = {cls: total/(count * NUM_CLASSES) for cls, count in class_counts.items()}


# train_sample, train_label = next(iter(train_dataset.take(1)))
# test_sample, test_label = next(iter(test_dataset.take(1)))

# print("Train Sample Shape:", train_sample.shape)
# print("Test Sample Shape:", test_sample.shape)



early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

model.fit(train_generator, batch_size = BATCH_SIZE, validation_data=val_generator, epochs=50, callbacks=[early_stop], class_weight=class_weights)

# test_dataset = test_dataset.batch(BATCH_SIZE)
# for images, labels in test_dataset.take(1):
#     print("Test Batch Shape:", images.shape)
#     print("Label Shape:", labels.shape)


# Evaluate on the test dataset

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model.save("optimized_asl_model")