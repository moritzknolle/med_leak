import os
from pathlib import Path

# set keras backend to jax and enable compilation caching
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]= "false"

import keras
import numpy as np
from absl import app, flags

from src.data_utils.dataset_factory import get_dataset
from src.train_utils.models.model_factory import get_model
from src.train_utils.training import train_and_eval, train_random_subset
from src.train_utils.utils import (
    MyCosineDecay,
    get_aug_fn,
)

FLAGS = flags.FLAGS
flags.DEFINE_integer("epochs", 100, "Number of training steps.")
flags.DEFINE_float("learning_rate", 2.0, "Learning rate.")
flags.DEFINE_float("weight_decay", 5e-3, "L2 weight decay.")
flags.DEFINE_float("momentum", 0.9, "Momentum parameter.")
flags.DEFINE_integer("batch_size", 512, "Batch size.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("log_wandb", True, "Whether to log metrics to weights & biases.")
flags.DEFINE_boolean(
    "ema", True, "Whether to use exponential moving average for parameters."
)
flags.DEFINE_string("model", "resnet1d_128", "Name of the model to use.")
flags.DEFINE_enum("lr_schedule", "cosine", ["constant", "cosine"], "LR schedule.")
flags.DEFINE_float(
    "lr_warmup",
    0.05,
    "Relative fraction of steps to perform linear learning rate warmup.",
)
flags.DEFINE_float(
    "decay_steps",
    1.0,
    "Relative fraction of total steps until learning rate is decayed to 1/10 times the original value. A value smaller than one means faster decay and likewise a bigger value leads to slower decay.",
)
flags.DEFINE_float("ema_decay", 0.995, "EMA decay.")
flags.DEFINE_integer("grad_accum_steps", 2, "Number of gradient accumulation steps.")
flags.DEFINE_float("dropout", 0.0, "Dropout rate.")
flags.DEFINE_enum(
    "augment",
    "none",
    ["trivial", "weak", "medium", "strong", "rotate", "none"],
    "What type of data augmentations strength to apply.",
)
flags.DEFINE_boolean(
    "mixed_precision",
    True,
    "Whether to perform mixed precision training to reduce training time.",
)
flags.DEFINE_boolean(
    "full_train_dataset",
    False,
    "Whether to use the full training dataset (Not the random subset otherwise used for leave-many-out re-training). This only has an effect when --eval_only is set to True.",
)
flags.DEFINE_bool("eval_only", True, "Whether to only evaluate the model.")
flags.DEFINE_integer(
    "n_runs", 200, "Number of leave-many-out re-training runs to perform."
)
flags.DEFINE_string(
    "save_root",
    "/home/moritz/data_fast/npy",
    "Path to root folder where the memmap files are stored.",
)
flags.DEFINE_float(
    "subset_ratio", 0.5, "Ratio of the training data to use for each re-training run."
)
flags.DEFINE_integer(
    "eval_views", 16, "Number of augmentations to query when saving train/test logits."
)
flags.DEFINE_string(
    "logdir",
    "./logs/ptb-xl/",
    "Path to logdir.",
)


def main(argv):
    np.random.seed(FLAGS.seed)
    if FLAGS.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    NUM_CLASSES = 5
    base_path = "/home/moritz/data_fast/npy/ptb-xl/ptb-xl"
    (x_train, y_train), (x_test, y_test) = get_dataset(
        dataset_name="ptb-xl",
        img_size=0,
        csv_root=Path("./data/csv"),
        data_root=Path("/home/moritz/data/physionet.org/files/ptb-xl/1.0.3/"),
        save_root=Path(FLAGS.save_root),
        get_numpy=True,
        load_from_disk=True,
        overwrite_existing=True,
    )
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    STEPS = len(x_train) // FLAGS.batch_size * FLAGS.epochs
    if not FLAGS.full_train_dataset:
        STEPS = int(STEPS * FLAGS.subset_ratio)

    def get_compiled_model():
        # create model, lr schedule and optimizer
        model = get_model(
            model_name=FLAGS.model,
            img_size=None,
            in_channels=12,
            num_classes=NUM_CLASSES,
            dropout=FLAGS.dropout,
            preprocessing_func=None,
        )
        schedule = MyCosineDecay(
            base_lr=FLAGS.learning_rate,
            steps=int(FLAGS.decay_steps * STEPS),
            relative_lr_warmup_steps=FLAGS.lr_warmup,
        )
        opt = keras.optimizers.SGD(
            learning_rate=(
                schedule if FLAGS.lr_schedule == "cosine" else FLAGS.learning_rate
            ),
            momentum=FLAGS.momentum,
            weight_decay=FLAGS.weight_decay,
            use_ema=FLAGS.ema,
            ema_momentum=FLAGS.ema_decay,
            gradient_accumulation_steps=FLAGS.grad_accum_steps,
        )
        # compile model
        model.compile(
            optimizer=opt,
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.CategoricalAccuracy(),
                keras.metrics.AUC(
                    multi_label=True, from_logits=True, name="macro_auroc"
                ),  # macro AUROC over all classes
            ],
        )
        return model

    def get_callbacks(is_ema: bool):
        callbacks = []
        if is_ema:
            callbacks += [keras.callbacks.SwapEMAWeights(swap_on_epoch=True)]
        return callbacks

    model = get_compiled_model()
    if FLAGS.eval_only:
        if not FLAGS.full_train_dataset:
            # randomly select 50% of the training data
            mask = np.random.binomial(1, 0.5, size=len(x_train)).astype(bool)
            subset_idcs = np.where(mask)[0]
            x_train = x_train[subset_idcs]
            y_train = y_train[subset_idcs]
        _, _, _, _ = train_and_eval(
            compiled_model=model,
            train_dataset=(x_train, y_train),
            test_dataset=(x_test, y_test),
            batch_size=FLAGS.batch_size,
            aug_fn=get_aug_fn(FLAGS.augment),
            augment=True if FLAGS.augment != "None" else False,
            epochs=FLAGS.epochs,
            target_metric="val_macro_auroc",
            callbacks=get_callbacks(FLAGS.ema),
            seed=FLAGS.seed,
            log_wandb=FLAGS.log_wandb,
            wandb_project_name="ptb-xl",
        )
    else:
        while True:
            try:
                train_random_subset(
                    compiled_model=model,
                    train_dataset=(x_train, y_train),
                    test_dataset=(x_test, y_test),
                    batch_size=FLAGS.batch_size,
                    aug_fn=get_aug_fn(FLAGS.augment),
                    augment=True if FLAGS.augment != "None" else False,
                    epochs=FLAGS.epochs,
                    target_metric="val_macro_auroc",
                    seed=FLAGS.seed,
                    logdir=Path(FLAGS.logdir),
                    n_total_runs=FLAGS.n_runs,
                    subset_ratio=FLAGS.subset_ratio,
                    n_eval_views=FLAGS.eval_views if FLAGS.augment != "none" else 1,
                    callbacks=get_callbacks(FLAGS.ema),
                    log_wandb=FLAGS.log_wandb,
                    wandb_project_name="ptb-xl",
                )
                model = get_compiled_model()
            except StopIteration:
                break


if __name__ == "__main__":
    app.run(main)
