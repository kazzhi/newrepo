import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot

# Load the trained model
model = tf.keras.models.load_model("asl_saved_model")  # or "asl_saved_model"


# Create a TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply optimizations to reduce model size
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enables post-training quantization
def representative_dataset():
    for _ in range(100):
        yield [np.random.rand(1, 64, 64, 1).astype(np.float32)]
        


converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # Use int8 quantization
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Convert the model
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open("asl_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted and saved as 'asl_model.tflite'")

