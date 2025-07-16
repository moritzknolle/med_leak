from functools import partial
from typing import Tuple

import keras


def get_small_cnn(
    img_size: Tuple[int, int] = (32, 32),
    in_channels: int = 3,
    num_classes: int = 10,
    base_filters: int = 32,
):
    """
    A small, fully convolutional neural network (around 1M parameters).
    """
    act = keras.layers.LeakyReLU(negative_slope=0.1)
    conv = partial(
        keras.layers.Conv2D,
        kernel_size=(3, 3),
        padding="same",
        activation=act,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
    )
    inputs = keras.Input(shape=(img_size[0], img_size[1], in_channels))
    x = inputs
    x = conv(base_filters)(x)
    x = conv(base_filters)(x)
    x = conv(base_filters * 2)(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = conv(base_filters * 2)(x)
    x = conv(base_filters * 2)(x)
    x = conv(base_filters * 4)(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = conv(base_filters * 4)(x)
    x = conv(base_filters * 4)(x)
    x = conv(base_filters * 8)(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = conv(num_classes, dtype="float32")(x)
    outputs = keras.layers.GlobalAveragePooling2D()(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
