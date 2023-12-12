import logging
import sys
from pathlib import Path

logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np


def generate_test_model():
    # Load MNIST dataset
    print("Loading MNIST dataset..")
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    print("Applying preprocessing to dataset ..")
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Define the model architecture
    print("Creating model ..")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
        ]
    )

    # Train the digit classification model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print("Training model for 5 epochs")
    model.fit(
        train_images, train_labels, epochs=5, validation_data=(test_images, test_labels)
    )

    return model

def representative_data_gen():
    mnist = tf.keras.datasets.mnist
    (train_images, _), _ = mnist.load_data()
    train_images = train_images.astype(np.float32) / 255.0
    for input_value in (
        tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100)
    ):
        print(input_value.dtype)
        # Model has only one input so each data point has one element.
        yield [input_value]

def convert_model_tflite_quant(model_path: Path):
    converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.experimental_new_converter = True

    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_data_gen
    quantized_tflite_model = converter.convert()
    return quantized_tflite_model

if __name__ == "__main__":
    output_path = sys.argv[1] 

    # Train or load the model
    tflite_models_dir = Path(output_path)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    model_file = tflite_models_dir/"mnist_model"
    if not model_file.exists():
        print("Creating MNIST classification model")
        mnist_model = generate_test_model()
        tf.saved_model.save(mnist_model, model_file)

    # Quantize the model
    mnist_tflite_quant = convert_model_tflite_quant(model_file)

    # Save the quantized model:
    tflite_model_quant_file = tflite_models_dir/"mnist_model_quant.tflite"
    tflite_model_quant_file.write_bytes(mnist_tflite_quant)
