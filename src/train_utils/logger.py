import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np

import wandb


class RetrainLogger:
    """
    A helper class for logging results and metrics of leave-many-out re-training runs.

    Args:
        wandb_run_id (int): The ID of the Weights & Biases run.
        logdir (Path): The directory where the logs will be saved.
        train_idcs (np.ndarray): The indices of the training data.

    Attributes:
        start_time (str): The start time of the retraining process.
        mac_address (str): The MAC address of the machine.
        config (wandb.config): The configuration settings.
        log_dir (Path): The directory where the logs will be saved.
        train_idcs (np.ndarray): The indices of the training data.

    Methods:
        save_metadata: Saves the metadata to a JSON file.
        log: Logs the training and evaluation data.
        get_mac_address: Retrieves the MAC address of the machine.
    """

    def __init__(
        self,
        logdir: Path,
        subset_idcs: np.ndarray,
        subset_mask: np.ndarray,
    ):
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.mac_address = self.get_mac_address()
        self.subset_idcs = subset_idcs
        self.subset_mask = subset_mask
        self.base_dir = logdir
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, metrics: dict):
        """
        Saves the metadata to a JSON file.

        Args:
            metrics (dict): The metrics to be saved.

        Returns:
            bool: True if the metadata is successfully saved, False otherwise.
        """
        self.config = dict(wandb.config)
        metrics = {k: float(v) for k, v in metrics.items()}
        try:
            info_dict = info_template
            info_dict["wandb_run_id"] = self.wandb_run_id
            info_dict["start_time"] = self.start_time
            info_dict["end_time"] = self.end_time
            info_dict["mac_address"] = self.mac_address
            info_dict["wandb_config"] = self.config
            info_dict["eval_metrics"] = metrics
            with open(self.log_dir / "info.json", "w") as f:
                json.dump(info_dict, f, indent=4)
        except Exception as e:
            print(e)
            return False
        return True

    def maybe_create_log_dir(self):
        if wandb.run is None:
            raise ValueError(
                "You must call wandb.init() before starting a RetrainExperiment()"
            )
        self.wandb_run_id = wandb.run.id
        self.log_dir = self.base_dir / str(self.wandb_run_id)
        if not self.log_dir.exists():
            print(f"... creating log_dir at {self.log_dir}")
            self.log_dir.mkdir(parents=True)

    def log(
        self,
        train_logits: np.ndarray,
        train_labels: np.ndarray,
        eval_logits: np.ndarray,
        eval_labels: np.ndarray,
        metrics: dict,
    ):
        """
        Logs the training and evaluation data.

        Args:
            train_logits (np.ndarray): The logits of the training data.
            train_labels (np.ndarray): The labels of the training data.
            eval_logits (np.ndarray): The logits of the evaluation data.
            eval_labels (np.ndarray): The labels of the evaluation data.
            metrics (dict): The metrics to be logged.

        Returns:
            bool: True if the data is successfully logged, False otherwise.
        """
        self.maybe_create_log_dir()
        self.end_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        assert len(train_logits) == len(train_labels) and len(train_logits) == len(
            self.subset_mask
        ), "train_logits, train_labels, and train_idcs must be the same length"
        assert train_logits.shape[1] == eval_logits.shape[1], "train_logits and eval_logits must have the same number of evaluations"
        try:
            np.save(self.log_dir / "train_logits.npy", train_logits)
            np.save(self.log_dir / "train_labels.npy", train_labels)
            np.save(self.log_dir / "eval_logits.npy", eval_logits)
            np.save(self.log_dir / "eval_labels.npy", eval_labels)
            np.save(self.log_dir / "subset_idcs.npy", self.subset_idcs)
            np.save(self.log_dir / "subset_mask.npy", self.subset_mask)
        except Exception as e:
            print(e)
            return False
        metadata_sucess = self.save_metadata(metrics)
        assert metadata_sucess, "Failed to save metadata"
        return True

    def log_simclr(
        self,
        train_features: List[np.ndarray],
        eval_features: List[np.ndarray],
        metrics: dict,
    ):
        """
        Logs the pair-wise cosine similarities of the training and evaluation data features.

        Args:
            train_cos_sims (np.ndarray): The pair-wise cosine similarities of the training data.
            train_pair_idcs (np.ndarray): The indices of the training data pairs.
            eval_cos_sims (np.ndarray): The pair-wise cosine similarities of the evaluation data.
            eval_pair_idcs (np.ndarray): The indices of the evaluation data pairs.
            metrics (dict): The metrics to be logged.

        Returns:
            bool: True if the data is successfully logged, False otherwise.
        """
        self.maybe_create_log_dir()
        self.end_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        try:
            joblib.dump(train_features, self.log_dir / "train_features.lib")
            joblib.dump(
                eval_features,
                self.log_dir / "eval_features.lib",
            )
            np.save(self.log_dir / "subset_idcs.npy", self.subset_idcs)
            np.save(self.log_dir / "subset_mask.npy", self.subset_mask)
        except Exception as e:
            print(e)
            return False
        metadata_sucess = self.save_metadata(metrics)
        assert metadata_sucess, "Failed to save metadata"
        return True

    def get_mac_address(self):
        """
        Retrieves the MAC address of the machine.

        Returns:
            str
        """
        try:
            mac_id = hex(uuid.getnode())
        except Exception as e:
            print(e)
            mac_id = "0000000000"
        return mac_id


info_template = {
    "wandb_run_id": 0,
    "start_time": "",
    "end_time": "",
    "mac_address": "",
    "wandb_config": {},
    "eval_metrics": {},
}
