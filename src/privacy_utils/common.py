from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_score(l_dir: Path, logit_transform_func: Callable, verbose: bool = False):
    """
    Loads and transforms MIA scores for a given log directory containing the results (predictions) of a model trained on a random training data subset.

    Args:
        l_dir: The path to the log directory.
    Returns:
        score: The logit transformed loss scores.
        subset_mask: A boolean mask indicating the subset of the data used for training.
        train_labels: The training labels.
    """
    try:
        logits = np.load(l_dir / "train_logits.npy")
        train_labels = np.load(l_dir / "train_labels.npy")
        subset_mask = np.load(l_dir / "subset_mask.npy")
        if verbose:
            print(
                f"Loaded logits of shape {logits.shape}, labels of shape {train_labels.shape}, and subset mask of shape {subset_mask.shape} from {l_dir}"
            )
        if not np.count_nonzero(np.isnan(logits)) == 0:
            raise Exception(f"Found NaNs logits in {l_dir}")
    except Exception as e:
        print(f"Could not load logits from {l_dir}, error message:\n {e}")
        return None
    if not logits.dtype == np.float32:
        print(
            f"WARNING: Logits are not of type float32, but {logits.dtype}. This might lead to numerical instabilities"
        )
        logits = logits.astype(np.float32)
    score = logit_transform_func(logits=logits, labels=train_labels)
    return score, subset_mask, train_labels


def load_score_disjoint(l_dir: Path, logit_transform_func: Callable):
    """
    Loads LiRA scores for a given log directory containing the results (predictions) of a model trained on data disjoint from the training data.

    Args:
        l_dir: The path to the log directory.
        logit_transform_func: The logit transform function to apply to the scores.
    Returns:
        score: The logit transformed loss scores.
        subset_mask: A boolean mask indicating the subset of the data used for training.
        train_labels: The training labels.
    """
    try:
        logits = np.load(l_dir / "eval_logits.npy")
        labels = np.load(l_dir / "eval_labels.npy")
    except Exception as e:
        print(f"Could not load logits from {l_dir}, error: {e}")
        return None
    score = logit_transform_func(logits=logits, labels=labels)
    return score, np.zeros(logits.shape[0], dtype=bool), labels


def aggregate_by_patient(aucs: np.ndarray, patient_ids: pd.Series):
    """
    Aggregates AUCs by patient ID to obtain a single MIA AUC per patient.
    Args:
        aucs: numpy array of shape (n_samples,) containing the record-level MIA AUC scores.
        patient_ids: numpy array of shape (n_samples,) containing the patient ID corresponding to each image.

    Returns:
        patient_auc: dictionary containing the aggregated patient-level MIA AUCs accessible through patient ID.
    """
    assert aucs.shape[0] == len(
        patient_ids
    ), f"Shapes do not match: {aucs.shape} vs {len(patient_ids)}"
    assert isinstance(
        patient_ids, pd.Series
    ), f"Expected patient_ids to be a pandas Series, but got {type(patient_ids)}"
    n_patients = patient_ids.nunique()
    auc_df = pd.DataFrame.from_dict({"aucs": aucs, "patient_id": patient_ids.values})
    aggregated = (
        auc_df.groupby("patient_id")["aucs"]
        .agg(mean="mean", std="std", max="max", count="count")
        .reset_index()
    )
    assert (
        len(aggregated) == n_patients
    ), f"Number of patients do not match dictionary size: {len(aggregated)} vs {n_patients}"
    return aggregated
