import json
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt #type: ignore
import numpy as np
import scipy
from absl import app, flags
from scipy.stats import norm

from mem_inf_stats import get_patient_col
from src.data_utils.dataset_factory import get_dataset
from src.privacy_utils.common import aggregate_by_patient
from src.privacy_utils.common import load_score
from src.privacy_utils.utils import convert_patientmask_to_recordmask
from src.privacy_utils.lira import (
    compute_scores,
    loss_logit_transform_multiclass,
    record_MIA_ROC_analysis,
)

FONT_SIZE = 7
plt.style.use("default")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force jax to use CPU
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "figure.figsize": (2, 2),
        "font.family": "sans serif",
        "font.sans-serif": "Inter",
        "axes.grid": False,
        "grid.alpha": 0.1,
        "axes.axisbelow": True,
        "figure.constrained_layout.use": True,
        # "pdf.fonttype": 42, # embed fonts in pdf
    }
)
FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_name", "ptb-xl", "Name of the dataset to plot data for ('ptb-xl' or '?').")
flags.DEFINE_integer("r_seed", 21, "Random seed.")
flags.DEFINE_float(
    "ylim_upper", 0.935, "upper y-limit for test performance metric plot"
)
flags.DEFINE_float("ylim_lower", 0.8, "lower y-limit for test performance metric plot")

PTB_LOG_DIRS = {
    #"eps1": "./logs/ptb-xl/dp/eps1",
    "eps10": "./logs/ptb-xl/dp/eps10",
    "eps100": "./logs/ptb-xl/dp/eps100",
    "eps1000": "./logs/ptb-xl/dp/eps1000",
    "epsinf": "./logs/ptb-xl/dp/epsinf",
}
COLORS = {
    "eps1": "#d6c4e9",
    "eps10": "#cab2e2",
    "eps100": "#996ac6",
    "eps1000": "#6b3a98",
    "epsinf": "#4e2b70",
}
NOISE_MULTIPLIERS = {
    "eps1": 14.901956181711261,
    "eps10": 2.217857337107472,
    "eps100": 0.6450520413560445,
    "eps1000": 0.2787387586165937,
    "epsinf": 0.00000001,
}

TRAIN_SIZE=9728

def compute_mu_uniform(
    steps: int, noise_multiplier: float, sample_rate: float
) -> float:
    """
    Compute mu from uniform subsampling.

    Args:
        steps: Number of steps taken
        noise_multiplier: Noise multiplier (sigma)
        sample_rate: Sample rate

    Returns:
        mu
    """

    c = sample_rate * np.sqrt(steps)
    return (
        np.sqrt(2)
        * c
        * np.sqrt(
            np.exp(noise_multiplier ** (-2)) * norm.cdf(1.5 / noise_multiplier)
            + 3 * norm.cdf(-0.5 / noise_multiplier)
            - 2
        )
    )

def calculate_auc_from_mu(mu:float) -> float:
    auc =norm.cdf(
        (mu) / np.sqrt(1**2 + 1**2)
    )  
    return auc

def calculate_dp_bound(logdir: Path):
    valid_dirs = [l for l in logdir.iterdir() if l.is_dir()]
    print(valid_dirs)
    with open(valid_dirs[0] / "info.json", "r") as f:
        config = json.load(f)["wandb_config"]
        clip_norm = config["clipping_norm"]
        noise_multiplier = config["noise_multiplier"]
        batch_size = config["batch_size"]


def get_perf_from_json(log_dir: Path, metric_name: str = "accuracy"):
    metric_vals = []
    valid_dirs = [l for l in log_dir.iterdir() if l.is_dir()]
    for subdir in valid_dirs:
        try:
            with open(subdir / "info.json", "r") as f:
                metrics = json.load(f)
                test_metrics = metrics["test_metrics"]
                for m_name, _ in test_metrics.items():
                    contains_numb = any(char.isdigit() for char in metric_name)
                    if contains_numb:
                        m_name_fixed = "_".join(
                            m_name.split("_")[-1]
                        )  # get rid off annoying number suffix that keras sometimes adds to metric names
                        metrics["test_metrics"][m_name_fixed] = metrics[
                            "test_metrics"
                        ].pop(m_name)
                metric_vals.append(metrics["test_metrics"][metric_name])
        except Exception as e:
            print(f"Encountered error loading {metric_name} from info.json: {e}")
            continue
    assert len(metric_vals) == len(
        valid_dirs
    ), f"Found {len(metric_vals)} values for {metric_name} but expected {len(valid_dirs)} \n found keys: {test_metrics.keys()}"
    return metric_vals


def main(argv):
    np.random.seed(FLAGS.r_seed)
    out_dir = Path(f"./figs/{FLAGS.dataset_name}")
    train_dataset, _ = get_dataset(
        dataset_name=FLAGS.dataset_name,
        img_size=[64, 64],
        csv_root=Path(FLAGS.csv_root),
        save_root=Path(FLAGS.save_root),
        data_root=Path(""),
        get_numpy=True,
        load_from_disk=True,
        overwrite_existing=True,
    )
    LOG_DIRS = PTB_LOG_DIRS if FLAGS.dataset_name == "ptb-xl" else 0
    test_metric_name = (
        "_val_macro_auroc" if FLAGS.dataset_name == "ptb-xl" else "_val_auroc"
    )
    logit_transform_func = partial(
        loss_logit_transform_multiclass, is_mimic_or_chexpert=False
    )
    load_func = partial(load_score, logit_transform_func=logit_transform_func)
    fig, axes = plt.subplots(
        1, 2, figsize=(3.5, 2), layout="tight", width_ratios=[1.7, 1.0]
    )
    test_metric_means, test_metric_stds = [], []
    for model_name, base_dir in LOG_DIRS.items():
        base_dir = Path(base_dir)
        logdirs = [d for d in base_dir.iterdir()]
        test_metric_vals = get_perf_from_json(base_dir, metric_name=test_metric_name)
        test_metric_means.append(np.mean(test_metric_vals))
        test_metric_stds.append(np.std(test_metric_vals))
        print(
            f"... average test {test_metric_name} for {model_name}: {np.mean(test_metric_vals)}"
        )
        scores, masks, _ = compute_scores(
            log_dirs=logdirs,
            load_score_func=load_func,
            multi_processing=True,
            threads=16,
        )
        masks = convert_patientmask_to_recordmask(
            patient_masks=masks,
            patient_ids=train_dataset.dataframe[
                get_patient_col(FLAGS.dataset_name)
            ].unique(),
            dataset=train_dataset,
            patient_id_col=get_patient_col(FLAGS.dataset_name),
        )
        _, _, aucs, _ = record_MIA_ROC_analysis(scores=scores, masks=masks)
        if FLAGS.dataset_name == "ptb-xl":
            patient_col = get_patient_col(FLAGS.dataset_name)
            patient_ids = train_dataset.dataframe[patient_col]
            patient_auc_df = aggregate_by_patient(aucs=aucs, patient_ids=patient_ids)
            aucs = patient_auc_df["max"]
        auc_bound = calculate_auc_from_mu(
            compute_mu_uniform(
                steps=(TRAIN_SIZE//1024)*200,
                noise_multiplier=NOISE_MULTIPLIERS[model_name],
                sample_rate=1024 / TRAIN_SIZE,
            )
        )
        res = scipy.stats.ecdf(aucs)
        conf = res.sf.confidence_interval(confidence_level=0.95)
        res.sf.plot(
            ax=axes[0],
            color=COLORS[model_name],
            label=f"{model_name}({np.mean(aucs):.2f}) [{auc_bound:.2f}]",
            alpha=0.8,
        )
        conf.low.plot(
            ax=axes[0], color=COLORS[model_name], linestyle="--", lw=1, alpha=0.8
        )
        conf.high.plot(
            ax=axes[0], color=COLORS[model_name], linestyle="--", lw=1, alpha=0.8
        )
        axes[0].axvline(
            auc_bound,
            color=COLORS[model_name],
            linestyle="dotted",
            lw=1,
            alpha=0.8,
        )
        print(
            f"Computed DP bound for {model_name} with mu={auc_bound:.3f} and noise_multiplier={NOISE_MULTIPLIERS[model_name]}.")
    axes[1].bar(
        range(len(LOG_DIRS)),
        test_metric_means,
        color=[COLORS[model_name] for model_name in LOG_DIRS.keys()],
        yerr=test_metric_stds,
        capsize=2,
    )
    axes[1].set_xticks(range(len(LOG_DIRS)))
    #axes[1].set_xticklabels(["10e1", "10e2", "10e3", r"$\infty$"], rotation=90)
    axes[1].set_xticklabels([])
    axes[1].set_xlabel(r"Privacy Budget ($\varepsilon$)")
    axes[1].set_ylabel("Diagnostic Performance")
    axes[1].grid(False)
    axes[1].set_ylim((FLAGS.ylim_lower, FLAGS.ylim_upper))
    axes[0].legend(loc="upper right", framealpha=0.5)
    axes[0].set_xlim((0.475, 1.01))
    ylim = 1 / len(aucs)
    axes[0].set_ylim((ylim, 1))
    axes[0].spines[["right", "top"]].set_visible(False)
    axes[1].spines[["right", "top"]].set_visible(False)
    xlabel = "MIA AUC (Patient-level)"
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("1 - Cumulative Probability")
    axes[0].set_yscale("log")
    plt.savefig(
        out_dir / f"{FLAGS.dataset_name}_dp_eSF.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    app.run(main)
