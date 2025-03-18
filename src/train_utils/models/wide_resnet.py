from functools import partial
from typing import Callable, List

import keras

# adapted from https://github.com/google/objax/blob/master/objax/zoo/wide_resnet.py

BN_MOM = 0.9
BN_EPS = 1e-5


def conv_args(kernel_size: int, nout: int):
    """Returns a list of arguments which are common to all convolutions.

    Args:
        kernel_size: size of convolution kernel (single number).
        nout: number of output filters.

    Returns:
        Dictionary with common convoltion arguments.
    """
    stddev = keras.ops.rsqrt(0.5 * kernel_size * kernel_size * nout)
    return dict(
        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=stddev),
        use_bias=False,
        padding="same",
    )


class WRNBlock(keras.layers.Layer):
    """WideResNet block."""

    def __init__(
        self,
        nin: int,
        nout: int,
        stride: int = 1,
        dropout: float = 0.0,
        norm: Callable = partial(
            keras.layers.BatchNormalization,
            momentum=BN_MOM,
            epsilon=BN_EPS,
            dtype="float32",
        ),
        act: Callable = keras.activations.relu,
    ):
        """Creates WRNBlock instance.

        Args:
            nin: number of input filters.
            nout: number of output filters.
            stride: stride for convolution and projection convolution in this block.
            norm: module which used as batch norm function.
        """
        super().__init__()
        if nin != nout or stride > 1:
            self.proj_conv = keras.layers.Conv2D(
                filters=nout, kernel_size=1, strides=stride, **conv_args(1, nout)
            )
        else:
            self.proj_conv = None

        self.norm_1 = norm()
        self.conv_1 = keras.layers.Conv2D(
            filters=nout, kernel_size=3, strides=stride, **conv_args(3, nout)
        )
        self.dropout = keras.layers.Dropout(dropout) if dropout > 0.0 else None
        self.norm_2 = norm()
        self.conv_2 = keras.layers.Conv2D(
            filters=nout, kernel_size=3, strides=1, **conv_args(3, nout)
        )
        self.act = act

    def call(self, inputs):
        x = inputs
        o1 = self.act(self.norm_1(x))
        y = self.conv_1(o1)
        if self.dropout:
            y = self.dropout(y)
        o2 = self.act(self.norm_2(y))
        z = self.conv_2(o2)
        return z + self.proj_conv(o1) if self.proj_conv else z + x


def get_wrn_general(
    in_channels: int,
    num_classes: int,
    img_size: int,
    blocks_per_group: List[int],
    width: int,
    norm: Callable = partial(
        keras.layers.BatchNormalization,
        momentum=BN_MOM,
        epsilon=BN_EPS,
        dtype="float32",
    ),
    act: Callable = keras.activations.relu,
    dropout: float = 0.0,
):
    """Builds a WideResNetGeneral Model instance.

    Args:
        in_channels: number of channels in the input image.
        nclass: number of output classes.
        blocks_per_group: number of blocks in each block group.
        width: multiplier to the number of convolution filters.
        norm: module which used as batch norm function.
    """
    widths = [
        int(v * width) for v in [16 * (2**i) for i in range(len(blocks_per_group))]
    ]
    n = 16
    inputs = keras.Input(shape=(img_size, img_size, in_channels))
    x = inputs
    x = keras.layers.Conv2D(filters=n, kernel_size=3, **conv_args(3, n))(x)
    for i, (block, width) in enumerate(zip(blocks_per_group, widths)):
        stride = 2 if i > 0 else 1
        x = WRNBlock(
            nin=n, nout=width, stride=stride, dropout=dropout, norm=norm, act=act
        )(x)
        for b in range(1, block):
            x = WRNBlock(
                nin=width, nout=width, stride=1, dropout=dropout, norm=norm, act=act
            )(x)
        n = width
    x = norm()(x)
    x = act(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(
        num_classes, kernel_initializer="glorot_normal", dtype="float32"
    )(x)

    return keras.Model(inputs=inputs, outputs=x)


def get_wide_resnet(
    img_size: int = 32,
    in_channels: int = 3,
    num_classes: int = 10,
    depth: int = 28,
    width: int = 2,
    norm: Callable = partial(
        keras.layers.BatchNormalization,
        momentum=BN_MOM,
        epsilon=BN_EPS,
        dtype="float32",
    ),
    act: str = "relu",
    dropout: float = 0.0,
):
    """Creates a WideResNet Model instance.

    Args:
        img_size: the image dimensionality.
        in_channels: number of channels in the input image.
        num_classes: number of output classes.
        depth: number of convolution layers. (depth-4) should be divisible by 6
        width: multiplier to the number of convolution filters.
        norm: module which used as batch norm function.
    """
    assert (depth - 4) % 6 == 0, "depth should be 6n+4"
    n = (depth - 4) // 6
    blocks_per_group = [n] * 3
    if act not in ["leaky_relu", "relu"]:
        raise ValueError(f"Activation {act} not found")
    act = (
        keras.layers.LeakyReLU(negative_slope=0.1)
        if act == "leaky_relu"
        else keras.activations.relu
    )
    return get_wrn_general(
        in_channels=in_channels,
        num_classes=num_classes,
        img_size=img_size,
        blocks_per_group=blocks_per_group,
        width=width,
        norm=norm,
        act=act,
        dropout=dropout,
    )
