# -*- coding: utf-8 -*-
"""emotion_detection_project.ipynb

Automatically generated by Colab.

**Emotion_Detection**

**Importing libraries**
"""

import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import cv2
import random
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import tensorflow as tf

print("numpy version:", np.__version__)
print("tensorflow version:", tf.__version__)

# Visualize sample images
images = glob("train/**/**")
plt.figure(figsize=(12, 12))
for i in range(9):
    image = random.choice(images)
    plt.subplot(331 + i)
    plt.imshow(cv2.imread(image, cv2.IMREAD_GRAYSCALE), cmap="gray")
    plt.axis("off")
plt.show()

# Preparing data for training
img_size = 48
batch_size = 64
epochs = 40
learning_rate = 0.001

# Data augmentation
datagen_train = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
train_generator = datagen_train.flow_from_directory(
    "train/",
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
)

datagen_validation = ImageDataGenerator(rescale=1.0 / 255)
validation_generator = datagen_validation.flow_from_directory(
    "test/",
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

# Define the CNN model
def Convolution(input_tensor, filters, kernel_size):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    return x


def Dense_f(input_tensor, nodes):
    x = Dense(nodes)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.3)(x)
    return x


def model_fer(input_shape):
    inputs = Input(input_shape)
    conv_1 = Convolution(inputs, 32, (3, 3))
    conv_2 = Convolution(conv_1, 64, (5, 5))
    conv_3 = Convolution(conv_2, 128, (3, 3))
    flatten = Flatten()(conv_3)
    dense_1 = Dense_f(flatten, 256)
    output = Dense(7, activation="softmax")(dense_1)

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(
        loss=["categorical_crossentropy"],
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


model = model_fer((img_size, img_size, 1))
model.summary()

# Callbacks for training
checkpoint = ModelCheckpoint(
    "model_weights.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
)
callbacks = [checkpoint, reduce_lr]

# Train the model
steps_per_epoch = int(train_generator.n / train_generator.batch_size)
validation_steps = int(validation_generator.n / validation_generator.batch_size)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
)

# Evaluate the model
model.evaluate(validation_generator)

# Plot loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.show()

# Save the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Visualizing activation maps
def visualize_activation_maps(model, image_path):
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing import image

    img = image.load_img(
        image_path, target_size=(img_size, img_size), color_mode="grayscale"
    )
    img_tensor = image.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)

    layer_outputs = [layer.output for layer in model.layers if "conv2d" in layer.name]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(img_tensor)

    layer_names = [layer.name for layer in model.layers if "conv2d" in layer.name]

    for layer_name, activation in zip(layer_names, activations):
        num_filters = activation.shape[-1]
        size = activation.shape[1]

        display_grid = np.zeros((size, size * num_filters))

        for i in range(num_filters):
            x = activation[0, :, :, i]
            x -= x.mean()
            x /= x.std() + 1e-5
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype("uint8")
            display_grid[:, i * size : (i + 1) * size] = x

        scale = 1.0 / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect="auto", cmap="viridis")


# Test visualization on a sample image
sample_image_path = random.choice(glob("test/**/**"))
visualize_activation_maps(model, sample_image_path)
