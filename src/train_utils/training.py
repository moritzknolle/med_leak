import fcntl
import functools
import os
import pickle
import time
from pathlib import Path
from typing import Callable, Tuple, Optional

import keras
import numpy as np
import tensorflow as tf
from absl import flags

import wandb

from .logger import RetrainLogger
from .utils import WandbLogger

AUTOTUNE = tf.data.experimental.AUTOTUNE


def prepare_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    aug_fn: Optional[Callable] = None,
):
    """
    Prepare a tf.data.Dataset from the given inputs and targets np arrays.
    Args:
        inputs: np.ndarray, input data
        targets: np.ndarray, target data
        batch_size: int, batch size
        shuffle: bool, whether to shuffle the dataset
        augment: bool, whether to augment the dataset
        aug_fn: Callable, augmentation function
    Returns:
        tf.data.Dataset, prepared dataset

    """
    print("... preparing dataset as tf.data.Dataset")
    # shape checks
    assert inputs.shape[0] == targets.shape[0], "Found mismatching number of samples"
    assert (
        inputs.max() != 255.0
    ), f"Make sure inputs are normalised appropriately, found max(inputs)={inputs.max()}"
    assert (
        inputs.dtype == keras.backend.floatx()
        and inputs.dtype == keras.backend.floatx()
    ), f"Found incorrects dtypes, expected {keras.backend.floatx()} and found {inputs.dtype} & {targets.dtype}"
    # prepare the dataset
    if inputs.base is None:
        print("... using in-memory dataset")
        ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    else:
        print("... using memmaped dataset")

        def gen():
            for i in range(len(inputs)):
                yield inputs[i], targets[i]

        ds = tf.data.Dataset.from_generator(
            gen,
            output_types=(inputs.dtype, targets.dtype),
            output_shapes=(inputs[0].shape, targets[0].shape),
        )

    if shuffle:
        ds = ds.shuffle(ds.cardinality(), reshuffle_each_iteration=True)
    
    if augment:
        ds = ds.map(
            lambda x, y: (aug_fn(x), y),
            num_parallel_calls=AUTOTUNE,
        )
    ds = ds.batch(batch_size)
    return ds.prefetch(buffer_size=AUTOTUNE)


def print_dataset_stats(input_arr: np.ndarray, target_arr: np.ndarray, split: str):
    """
    Print dataset statistics for the given input and target arrays.
    Args:
        input_arr: np.ndarray, input data
        target_arr: np.ndarray, target data
        split: str, split name
    Returns:
        None
    """

    print(f"... {split} dataset stats")
    print(f"    input shape: {input_arr.shape} ({input_arr.dtype})")
    if input_arr.base is not None:
        print(
            f"    input: min={input_arr.min():.2f} max={input_arr.max():.2f} mean={input_arr.mean():.2f} std={input_arr.std():.2f}"
        )
    print(f"    target shape: {target_arr.shape} ({target_arr.dtype})")
    if target_arr.base is not None:
        print(
            f"    target: min={target_arr.min():.2f} max={target_arr.max():.2f} mean={target_arr.mean():.2f} std={target_arr.std():.2f}"
        )


def train_and_eval(
    compiled_model: keras.Model,
    train_dataset: Tuple[np.ndarray, np.ndarray],
    test_dataset: Tuple[np.ndarray, np.ndarray],
    batch_size: int,
    aug_fn: Callable,
    augment: bool,
    epochs: int,
    seed: int,
    target_metric: str,
    callbacks: list = [],
    ckpt_file_path: Path = Path("./tmp/ckpt/"),
    track_data_stats: bool = True,
    overfit: bool = False,
    wandb_project_name: str = "",
    log_wandb: bool = False,
    verbose: bool = True,
) -> Tuple[keras.Model, dict, dict, bool]:
    """
    Train and evaluate a model using the given datasets and training parameters.

    Args:
        compiled_model: keras.Model, compiled model
        train_dataset: Tuple[np.ndarray, np.ndarray], training dataset
        test_dataset: Tuple[np.ndarray, np.ndarray], testing dataset
        batch_size: int, batch size
        aug_fn: Callable, augmentation function
        augment: bool, whether to augment training data
        epochs: int, number of epochs to train
        seed: int, random seed
        callbacks: list, list of callbacks
        track_data_stats: bool, whether to track data statistics
        overfit: bool, whether to overfit on first batch
        wandb_project_name: str, wandb project name
        log_wandb: bool, whether to log to wandb
        verbose: bool, whether to print verbose logs

    Returns:
        Tuple[keras.Model, dict, dict], trained model, training history, test metrics

    """
    keras.utils.set_random_seed(seed)
    print(f"... starting training using keras backend: {keras.backend.backend()}")
    if keras.backend.backend() == "jax":
        import jax

        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    if log_wandb:
        if wandb_project_name == "":
            raise ValueError("Must provide a valid wandb project name")
        wandb.init(project=f"{wandb_project_name}")
        wandb.config.update(flags.FLAGS)
        wandb.config.update({"run_seed": seed})
        wandb.config.update({"epochs": epochs})
    n_train = len(train_dataset[0])
    n_test = len(test_dataset[0])
    is_memmap = train_dataset[0].base is not None
    (
        print_dataset_stats(train_dataset[0], train_dataset[1], "train")
        if train_dataset[0].base is None
        else 0
    )
    (
        print_dataset_stats(test_dataset[0], test_dataset[1], "test")
        if test_dataset[0].base is None
        else 0
    )
    print(
        f"... training for {epochs} epochs ({epochs*(len(train_dataset[0])//batch_size)} steps)"
    )

    # prepare training and test datasets
    train_dataset = prepare_dataset(
        inputs=train_dataset[0],
        targets=train_dataset[1],
        batch_size=batch_size,
        shuffle=True,
        augment=augment,
        aug_fn=aug_fn,
    )
    test_dataset = prepare_dataset(
        inputs=test_dataset[0],
        targets=test_dataset[1],
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        aug_fn=None,
    )
    # save a few random images to ./tmp for debugging purposes
    if track_data_stats:
        tmp_dir = Path("./tmp/imgs")
        tmp_dir.mkdir(exist_ok=True)
        train_batch = next(iter(train_dataset.take(1)))
        test_batch = next(iter(test_dataset.take(1)))
        random_idcs = np.random.randint(0, train_batch[0].shape[0], size=15)
        for i, idc in enumerate(random_idcs):
            if train_batch[0].shape[-1] in [1,3]:
                tf.keras.preprocessing.image.save_img(
                    f"{tmp_dir}/train_{i}.png", train_batch[0][i]
                )
                tf.keras.preprocessing.image.save_img(
                    f"{tmp_dir}/test_{i}.png", test_batch[0][i]
                )
    if overfit:

        def first_batch_only(dataset: tf.data.Dataset):
            batch = next(iter(dataset))
            total_steps = len(dataset)
            i = 0
            while i < total_steps:
                yield batch
                i += 1

        print("... delibaretely overfitting on first batch. Careful!")
        train_dataset = first_batch_only(train_dataset)
    print(f"... training on {n_train} samples, evaluating on {n_test} samples")

    start_time = time.time()

    # model checks
    assert isinstance(compiled_model, keras.Model), "Model must be a keras model"
    assert (
        compiled_model.compiled
    ), f"Model must be compiled!, compilation status: {compiled_model.compiled}"
    print(compiled_model.summary())
    callbacks.append(keras.callbacks.TerminateOnNaN())
    wandb_callbacks = [WandbLogger(model=compiled_model)] if log_wandb else []
    if ckpt_file_path is not None:
        if log_wandb:
            # if using wandb, save model weights with wandb run id to avoid overwriting issues with multiple parallel runs
            ckpt_file_path = ckpt_file_path / f"model_{wandb.run.id}.weights.h5"
        else:
            ckpt_file_path = ckpt_file_path / "model.weights.h5"
        print("... creating model checkpoint callback at ", ckpt_file_path)
        ckpt_callback = keras.callbacks.ModelCheckpoint(
            filepath=ckpt_file_path,
            save_weights_only=True,
            save_best_only=True,
            monitor=target_metric,
            mode="max",
            verbose=1,
        )
        callbacks.append(ckpt_callback)
    callbacks = wandb_callbacks + callbacks
    try:
        training_history = compiled_model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=test_dataset,
            verbose=verbose,
            callbacks=callbacks,
            steps_per_epoch=n_train // batch_size if is_memmap else None,
            validation_steps=n_test // batch_size if is_memmap else None,
        )
    except KeyboardInterrupt:
        if ckpt_file_path is not None:
            print(f"... deleting checkpoint file: {ckpt_file_path}")
            del ckpt_callback
            os.remove(ckpt_file_path)
    failed = (
        compiled_model.stop_training
    )  # this will evaluate to True if NaN loss was encountered during training
    training_time = (time.time() - start_time) / 60  # training time in minutes
    print(f"... finished training in {training_time} mins")
    # evaluation
    if ckpt_file_path is not None:
        # load best weights
        compiled_model.load_weights(ckpt_file_path)
        print(f"... deleting checkpoint file: {ckpt_file_path}")
        del ckpt_callback
        os.remove(ckpt_file_path)
    test_metrics = compiled_model.evaluate(
        test_dataset, verbose=verbose, return_dict=True
    )
    print(f"... test metrics: {test_metrics}")
    if log_wandb:
        test_metrics = {f"_val_{k}": v for k, v in test_metrics.items()} | {
            "training_time": training_time
        }
        wandb.log(test_metrics)
    return compiled_model, training_history, test_metrics, failed


def generate_masks(
    n_runs: int,
    dataset_size: int,
    subset_ratio: float,
    seed: int,
):
    """
    Generate random subset masks for the given number of runs and dataset size.
    Adapted from: https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py

    Args:
        n_runs: int, number of runs
        dataset_size: int, size of the dataset
        subset_ratio: float, ratio of subset
        seed: int, random seed
    Returns:
        np.ndarray, random subset masks
    """
    np.random.seed(seed)
    super_mask = np.random.uniform(0.0, 1.0, size=(n_runs, dataset_size))
    order = super_mask.argsort(0)
    super_mask = order < int(subset_ratio * n_runs)
    if super_mask.shape[1] != dataset_size:
        raise ValueError(
            f"Found mismatching dataset size: {super_mask.shape[1]} != {dataset_size}"
        )
    if subset_ratio == 1.0:
        super_mask = np.ones_like(super_mask)
    return super_mask


def train_random_subset(
    compiled_model: keras.Model,
    train_dataset: Tuple[np.ndarray, np.ndarray],
    test_dataset: Tuple[np.ndarray, np.ndarray],
    batch_size: int,
    aug_fn: Callable,
    augment: bool,
    epochs: int,
    seed: int,
    target_metric: str,
    logdir: Path,
    ckpt_file_path: Path = Path("./tmp/ckpt/"),
    n_total_runs: int = 150,
    subset_ratio: float = 0.5,
    n_eval_views: int = 1,
    augment_test: bool = False,
    callbacks: list = [],
    track_data_stats: bool = True,
    overfit: bool = False,
    wandb_project_name: str = "",
    log_wandb: bool = False,
    verbose: bool = True,
):
    """
    Train a model on a random subset of train_dataset. Logs to wandb and saves logits and corresponding labels of train and test dataset to logdir.
    Completely standard Training.

    Args:
        compiled_model: keras.Model, compiled model
        train_dataset: Tuple[np.ndarray, np.ndarray], training dataset
        test_dataset: Tuple[np.ndarray, np.ndarray], testing dataset
        batch_size: int, batch size
        aug_fn: Callable, augmentation function
        augment: bool, whether to augment training data
        epochs: int, number of epochs to train
        seed: int, random seed
        logdir: Path, directory to save logs
        n_total_runs: int, total number of runs
        subset_ratio: float, ratio of subset
        n_eval_views: int, number of views for evaluation
        augment_test: bool, whether to augment test data
        callbacks: list, list of callbacks
        track_data_stats: bool, whether to track data statistics
        overfit: bool, whether to overfit on first batch
        wandb_project_name: str, wandb project name
        log_wandb: bool, whether to log to wandb
        verbose: bool, whether to print verbose logs

    Returns:
        None
    """
    assert logdir is not None, "Must provide a logdir to save logs"
    assert isinstance(logdir, Path), "logdir must be a pathlib.Path object"
    keras.utils.clear_session(
        free_memory=True
    )  # clear keras session from a potential previous run
    logdir.mkdir(parents=True, exist_ok=True)
    # check if supermask file and indices file exists
    super_mask_path = logdir / "super_mask.npy"
    valid_idcs_path = logdir / "valid_idcs.pkl"
    completed_idcs_path = logdir / "completed_idcs.pkl"
    # if exists, load it, pick a subset mask that hasn't been picked and indicate that it has been picked
    if (
        super_mask_path.exists()
        and valid_idcs_path.exists()
        and completed_idcs_path.exists()
    ):
        with open(valid_idcs_path, "r+b") as f:
            with open(completed_idcs_path, "r+b") as f2:
                # lock file to prevent race condition
                fcntl.flock(f, fcntl.LOCK_EX)
                super_mask = np.load(super_mask_path)
                valid_idcs = pickle.load(f)
                completed_idcs = pickle.load(f2)
                if len(valid_idcs) == 0:
                    print("... No more valid subset masks left.")
                    if len(completed_idcs) >= n_total_runs:
                        print("... done training. Exiting")
                        raise StopIteration
                    else:
                        print("... checking for failed runs")
                        all_idcs = list(range(n_total_runs))
                        missing_idcs = list(set(all_idcs) - set(completed_idcs))
                        if len(missing_idcs) == 0:
                            print("... no failed runs found. Exiting")
                            raise StopIteration
                        else:
                            print(
                                f"... found {len(missing_idcs)} failed runs, retrying"
                            )
                            valid_idcs = missing_idcs
                else:
                    print(
                        f"... processing run {valid_idcs[0]}, {len(valid_idcs)} runs left, {len(completed_idcs)} runs completed"
                    )
                next = valid_idcs.pop(0)
                subset_mask = super_mask[next]
                f.seek(0)
                f.truncate()
                # unlock file
                fcntl.flock(f, fcntl.LOCK_UN)
                pickle.dump(valid_idcs, f)
    # otherwise, generate new supermask and indices file
    else:
        super_mask = generate_masks(
            n_runs=n_total_runs,
            dataset_size=len(train_dataset[0]),
            subset_ratio=subset_ratio,
            seed=seed,
        )
        valid_idcs = list(range(n_total_runs))
        completed_idcs: list[int] = []
        next = valid_idcs.pop(0)
        subset_mask = super_mask[next]
        np.save(super_mask_path, super_mask)
        with open(valid_idcs_path, "w+b") as f:
            pickle.dump(valid_idcs, f)
        with open(completed_idcs_path, "w+b") as f:
            pickle.dump(completed_idcs, f)
    assert len(subset_mask) == len(
        train_dataset[0]
    ), "Subset mask must match dataset size"
    subset_idcs = np.nonzero(subset_mask)[0]
    train_inputs, train_targets = train_dataset
    sub_train_inputs = train_inputs[subset_idcs]
    sub_train_targets = train_targets[subset_idcs]
    assert len(sub_train_inputs) == len(sub_train_targets), f"Mismatching input shapes: {len(sub_train_inputs)} != {len(sub_train_targets)}"
    if len(sub_train_inputs) >= len(train_inputs):
        print("... Warning. Subset is not smaller than full dataset!")
    assert (
        len(sub_train_inputs) == subset_mask.sum()
    ), f"Mismatch between subset size and mask: {len(sub_train_inputs)} != {subset_mask.sum()}"
    assert (
        sub_train_inputs.shape[0] == sub_train_targets.shape[0]
    ), "Mismatching input shapes"
    print(
        train_inputs.shape,
        train_targets.shape,
        sub_train_inputs.shape,
        sub_train_targets.shape,
        test_dataset[0].shape,
        test_dataset[1].shape,
    )
    if not isinstance(logdir, Path) and logdir is not None:
        logdir = Path(logdir)
    logger = RetrainLogger(
        logdir=logdir, subset_idcs=subset_idcs, subset_mask=subset_mask
    )
    exception_raised = False
    try:
        compiled_model, training_history, eval_metrics, run_failed = train_and_eval(
            compiled_model=compiled_model,
            train_dataset=(sub_train_inputs, sub_train_targets),
            test_dataset=test_dataset,
            batch_size=batch_size,
            aug_fn=aug_fn,
            augment=augment,
            epochs=epochs,
            seed=seed,
            target_metric=target_metric,
            ckpt_file_path=ckpt_file_path,
            callbacks=callbacks,
            track_data_stats=track_data_stats,
            overfit=overfit,
            wandb_project_name=wandb_project_name,
            log_wandb=log_wandb,
            verbose=verbose,
        )
    except Exception as e:
        print(f"Error: {e}")
        exception_raised = True
    if exception_raised or run_failed:
        print(f"... run failed, adding {next} back to valid indices")
        # re-add supermask index to valid indices if training failed
        with open(valid_idcs_path, "r+b") as f:
            # lock file to prevent
            fcntl.flock(f, fcntl.LOCK_EX)
            valid_idcs = pickle.load(f)
            valid_idcs.append(next)
            f.seek(0)
            f.truncate()
            # unlock file
            fcntl.flock(f, fcntl.LOCK_UN)
            pickle.dump(valid_idcs, f)
    else:
        # save logits and labels of train as well as test set
        if log_wandb:
            assert len(train_inputs) != len(
                sub_train_inputs
            ), "Random subset must be smaller than full dataset"
            train_ds = prepare_dataset(
                inputs=train_inputs,  # full training dataset
                targets=train_targets,
                batch_size=batch_size,
                shuffle=False,
                augment=augment,
                aug_fn=aug_fn,
            )
            test_ds = prepare_dataset(
                inputs=test_dataset[0],
                targets=test_dataset[1],
                batch_size=batch_size,
                shuffle=False,
                augment=augment_test,
                aug_fn=aug_fn,
            )
            # set dtype policy to float32
            keras.config.set_dtype_policy("float32")
            # save logits and labels
            print("... fininished training, saving logits and labels")
            if n_eval_views > 1:
                train_logits, test_logits = [], []
                for i in range(n_eval_views):
                    train_logits.append(compiled_model.predict(train_ds))
                    test_logits.append(compiled_model.predict(test_ds))
                train_logits = np.stack(train_logits, axis=1)
                test_logits = np.stack(test_logits, axis=1)
            else:
                train_logits = compiled_model.predict(train_ds)
                test_logits = compiled_model.predict(test_ds)
            print(train_logits.shape, test_logits.shape)
            success = logger.log(
                train_logits=train_logits,
                train_labels=train_targets,
                eval_logits=test_logits,
                eval_labels=test_dataset[1],
                metrics=eval_metrics,
            )
            if success:
                # add current index to completed indices
                with open(completed_idcs_path, "r+b") as f2:
                    print(f"... succesfully completed run: {next}")
                    # lock file to prevent race condition
                    fcntl.flock(f2, fcntl.LOCK_EX)
                    # re-load completed indices in case another process has modified it
                    completed_idcs = pickle.load(f2)
                    completed_idcs.append(next)
                    f2.seek(0)
                    f2.truncate()
                    # unlock file
                    fcntl.flock(f2, fcntl.LOCK_UN)
                    pickle.dump(completed_idcs, f2)
    wandb.run.finish()
