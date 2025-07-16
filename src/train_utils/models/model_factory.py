from functools import partial
from typing import Callable, Optional, Tuple

import keras

from .small_models import get_small_cnn
from .vit.vision_transformer import vit_b8, vit_b16, vit_b32, vit_l16, vit_l32
from .wide_resnet import get_wide_resnet
from .resnet_1d import build_1d_resnet, build_tabular_resnet


def get_model(
    model_name: str,
    img_size: Tuple[int, int],
    in_channels: int,
    num_classes: int,
    dropout=0.0,
    preprocessing_func: Optional[Callable] = None,
):
    """
    Helper function that returns a function which creates the respective model given the model name.
    Args:
        model_name: str, name of the model
        img_size: int, size of the input image
        in_channels: int, number of channels of the input image
        num_classes: int, number of classes
        dropout: float, dropout rate
        preprocessing_func: callable, preprocessing function to apply to the input image. Only effective for ViT models.
    Returns:
        Callable: function that returns the corresponding model
    """
    if model_name == "small_cnn":
        if dropout != 0.0:
            raise ValueError("Small CNN does not support dropout")
        model = get_small_cnn(
            img_size=img_size, in_channels=in_channels, num_classes=num_classes
        )
    elif model_name.split("_")[0] == "wrn":
        depth = int(model_name.split("_")[1])
        width = int(model_name.split("_")[2])
        model = get_wide_resnet(
            depth=depth,
            width=width,
            img_size=img_size,
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif model_name == "resnet50":
        backbone = keras.applications.ResNet50(
            weights=None,
            include_top=False,
            input_shape=(img_size[0], img_size[1], in_channels),
            classes=num_classes,
        )
        model = keras.Sequential(
            [
                backbone,
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(num_classes, dtype="float32"),
            ]
        )
    elif model_name == "resnet50_imagenet":
        backbone = keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size[0], img_size[1], 3),
            classes=num_classes,
        )
        model = keras.Sequential(
            [
                backbone,
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(num_classes, dtype="float32"),
            ]
        )
    elif model_name == "densenet121":
        backbone = keras.applications.DenseNet121(
            weights=None,
            include_top=False,
            input_shape=(img_size[0], img_size[1], in_channels),
            classes=num_classes,
        )
        model = keras.Sequential(
            [
                backbone,
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(num_classes, dtype="float32"),
            ]
        )
    elif model_name == "densenet121_imagenet":
        backbone = keras.applications.DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size[0], img_size[1], 3),
            classes=num_classes,
        )
        model = keras.Sequential(
            [
                backbone,
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(num_classes, dtype="float32"),
            ]
        )
    elif model_name.split("_")[0] == "resnet1d" or model_name == "resnet1d":
        nb_feature_maps = int(model_name.split("_")[1]) if len(model_name.split("_")) > 1 else 64
        model = build_1d_resnet(nb_classes=num_classes, input_shape=(1_000, in_channels), nb_feature_maps=nb_feature_maps)
    elif model_name.split("_")[0] == "tabresnet":
        _, width, n_blocks = model_name.split("_")
        width = int(width)
        n_blocks_int = int(n_blocks)
        model = build_tabular_resnet(
            input_shape=(in_channels,),
            width=width,
            depth=n_blocks_int,
            dropout_rate=dropout,
            num_classes=num_classes,
        )
        
    elif model_name.split("_")[0] == "vit":
        if len(model_name.split("_")) != 3:
            raise ValueError(
                f"Invalid size {model_name}, expected format 'vit_[model_size]_[patch_size]'"
            )
        model_size = model_name.split("_")[1]
        patch_size = int(model_name.split("_")[2])
        if model_size == "b" and patch_size == 8:
            model = vit_b8(
                image_size=img_size,
                activation="linear",
                pretrained=True,
                include_top=True,
                classes=num_classes,
                pretrained_top=False,
            )
        elif model_size == "b" and patch_size == 16:
            model = vit_b16(
                image_size=img_size,
                activation="linear",
                pretrained=True,
                include_top=True,
                classes=num_classes,
                pretrained_top=False,
            )
        elif model_size == "b" and patch_size == 32:
            model = vit_b32(
                image_size=img_size,
                activation="linear",
                pretrained=True,
                include_top=True,
                classes=num_classes,
                pretrained_top=False,
            )
        elif model_size == "l" and patch_size == 16:
            model = vit_l16(
                image_size=img_size,
                activation="linear",
                pretrained=True,
                include_top=True,
                classes=num_classes,
                pretrained_top=False,
            )
        elif model_size == "l" and patch_size == 32:
            model = vit_l32(
                image_size=img_size,
                activation="linear",
                pretrained=True,
                include_top=True,
                classes=num_classes,
                pretrained_top=False,
            )
        else:
            raise ValueError(f"Invalid ViT variant {model_name}, model variant: {model_size}, patch size: {patch_size}")
        if in_channels == 1:
            if preprocessing_func is None:
                raise ValueError(
                    "Preprocessing function must be provided for grayscale images with ViT"
                )
    else:
        raise ValueError(f"Model {model_name} not found")
    if preprocessing_func is not None:
        final_model = keras.Sequential(
            [
                keras.layers.Input(shape=(img_size[0], img_size[1], in_channels)),
                keras.layers.Lambda(preprocessing_func, name="preprocessing"),
                model,
            ]
        )
    else:
        final_model = model
    return final_model
