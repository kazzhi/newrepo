import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import collections
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
# import matplotlib.pyplot as plt



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
    color_mode="rgb",
    shuffle=False
).map(preprocess_image)

#test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y))


def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """Optimized CNN model for ESP32-S3 with ESP-NN support."""
    base_model = keras.applications.MobileNetV2(
        input_shape=(64, 64, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze the base model

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax")
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
#datagen.fit(X_train)
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

for image, label in test_dataset.take(1):
    print("Image shape:", image.shape)
    print("Label:", label)

class_counts = collections.Counter(y_train)
total = sum(class_counts.values())
class_weights = {cls: total/(count * NUM_CLASSES) for cls, count in class_counts.items()}



# for images, labels in test_dataset.take(1):
#     for i in range(5):  # Inspect 5 samples
#         plt.imshow(images[i].numpy().squeeze(), cmap='gray')
#         plt.title(f"Label: {labels[i].numpy()}")
#         plt.show()

# Include in model.fit()
# Train the model
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

model.fit(train_generator, batch_size = BATCH_SIZE, validation_data=val_generator, epochs=50, callbacks=[early_stop], class_weight=class_weights)
test_dataset = test_dataset.shuffle(buffer_size=len(test_dataset))
# Evaluate on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model.save("optimized_asl_model")