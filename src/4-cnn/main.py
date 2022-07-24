#!/usr/bin/env python3


# Copyright 2019 The TensorFlow Authors (original code)
# Copyright 2022 Berke Kocaoğlu (derivative)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
import numpy as np


def cnn_simple():
    from tensorflow.keras import datasets, layers, models, losses, optimizers
    import matplotlib.pyplot as plt

    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = datasets.cifar10.load_data()

    # normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = models.Sequential()
    # relu = R(z) = max(0, z)
    model.add(layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation=tf.nn.relu))

    model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation=tf.nn.relu))
    model.add(layers.Dense(10))

    model.summary()

    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_data=(test_images, test_labels),
    )

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(test_acc)


def cnn_advanced():
    from tensorflow.keras import Model, datasets, losses, optimizers, metrics, models
    from tensorflow.keras.layers import Dense, Flatten, Conv2D

    mnist = datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # add a channels dimension
    x_train = x_train[..., tf.newaxis].astype(np.float32)
    x_test = x_test[..., tf.newaxis].astype(np.float32)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    )

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    class CustomModel(Model):
        def __init__(self):
            super(CustomModel, self).__init__()
            self.conv1 = Conv2D(32, 3, activation=tf.nn.relu)
            self.flatten = Flatten()
            self.d1 = Dense(128, activation=tf.nn.relu)
            self.d2 = Dense(10)

        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

    model = CustomModel()

    loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = optimizers.Adam()

    train_loss = metrics.Mean(name="train_loss")
    train_accuracy = metrics.SparseCategoricalAccuracy(name="train_accuracy")

    test_loss = metrics.Mean(name="test_loss")
    test_accuracy = metrics.SparseCategoricalAccuracy(name="test_accuracy")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behaviour during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behaviour during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 60  # starts overfitting at about 50

    for epoch in range(EPOCHS):
        # reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {train_loss.result()}, "
            f"Accuracy: {train_accuracy.result() * 100}, "
            f"Test Loss: {test_loss.result()}, "
            f"Test Accuracy: {test_accuracy.result() * 100}"
        )

    model.save("custom_model", save_format="tf")
    model.summary()
    model = models.load_model("custom_model")
    model.summary()
    model.compile()


def get_func():
    while True:
        match input(
            "Choose method [simple (s, 1), advanced (a, 2)]:\n> "
        ).strip().lower():
            case "simple" | "s" | "1":
                return cnn_simple
            case "advanced" | "a" | "2":
                return cnn_advanced
            case _:
                continue


def main():
    get_func()()


if __name__ == "__main__":
    main()
