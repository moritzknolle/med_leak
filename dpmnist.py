import os
from absl import app # type: ignore

os.environ["KERAS_BACKEND"] = "jax"
from jax_privacy.keras import keras_api  # type: ignore
import keras # type: ignore
from keras import layers # type: ignore
import numpy as np

num_classes = 10
input_shape = (28, 28, 1)


def get_model():
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])


def load_data():
    """Loads the MNIST data and returns the train and test sets."""
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # convert class vectors to "one hot encoding"
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


def main(_):
    # Marker to insert the main part of the example into ReadTheDocs.
    # [START example]
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"train data shape: {x_train.shape}, {y_train.shape}")
    print(f"test data shape: {x_test.shape}, {y_test.shape}")
    model = get_model()

    epsilon = 5.0
    delta = 1e-5
    batch_size = 1_000
    epochs = 25
    train_size = len(x_train)
    dp = True
    clipping_norm = 0.5

    if dp:
        params = keras_api.DPKerasConfig(
            epsilon=epsilon,
            delta=delta,
            clipping_norm=clipping_norm,
            batch_size=batch_size,
            gradient_accumulation_steps=5,
            train_steps=epochs * (train_size // batch_size),
            train_size=train_size,
            seed=0,
        )
        model = keras_api.make_private(model, params)
        print(
            f"DP training:{epsilon=} {delta=} {clipping_norm=} {batch_size=} "
            f" {epochs=} {train_size=}"
        )
    else:
        print("Non-DP training")
    model.compile(
        loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(gradient_accumulation_steps=5, learning_rate=5e-3), metrics=["accuracy"]
    )
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
    )
    # [END example]
    print("DP: expected train accuracy: ~96%, val accuracy: ~92%")
    print("Non-DP: expected train accuracy: ~98%, val accuracy: ~98%")


if __name__ == "__main__":
    app.run(main)