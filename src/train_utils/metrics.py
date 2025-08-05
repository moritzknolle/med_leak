from typing import Dict

import jax, keras # type: ignore
import numpy as np
import sklearn.metrics as skmetrics
from scipy.interpolate import interp1d

from ..data_utils.constants import (
    CXP_CHALLENGE_LABELS,
    CXP_CHALLENGE_LABELS_IDX,
    CXR_LABELS,
)


def tpr_at_fpr(
    targets: np.ndarray,
    preds: np.ndarray,
    fpr_of_interest: float,
    n_interp_samples=10_000,
):
    assert preds.shape == targets.shape, f"{preds.shape} != {targets.shape}"
    if fpr_of_interest < 0.01:
        # for numerical stability we use log-space interpolation
        x = np.logspace(-8, 0, n_interp_samples)
    else:
        x = np.linspace(0, 1, n_interp_samples)
    x = np.flip(x)
    fpr, tpr, thresholds = skmetrics.roc_curve(targets, preds)
    tpr_interp = interp1d(fpr, tpr)(x)  # interpolate fpr at tpr
    idx = np.abs(x - fpr_of_interest).argmin()
    return tpr_interp[idx]
