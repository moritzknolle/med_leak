from pathlib import Path

import joblib
import matplotlib.patches as mpat
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

from src.colors import (chex_color, embed_color, fairvision_color,
                              fitz_color, mimic_color, ptb_xl_color, mimic_iv_ed_color)
from src.privacy_utils.plot_utils import mia_auc_esf_plot

FONT_SIZE = 7
plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "figure.figsize": (2.4, 2.15),
        "font.family": "sans serif",
        "font.sans-serif": "Inter",
        "axes.grid": False,
        "grid.alpha": 0.1,
        "axes.axisbelow": True,
        "figure.constrained_layout.use": True,
        # "pdf.fonttype":42,
    }
)

FIGDIRS = {
    "chexpert": "./figs/chexpert",
    "mimic": "./figs/mimic",
    "fitzpatrick": "./figs/fitzpatrick",
    "fairvision": "./figs/fairvision",
    "embed": "./figs/embed",
    "ptb-xl": "./figs/ptb-xl",
    "mimic-iv-ed": "./figs/mimic-iv-ed",
}
FLAGS = flags.FLAGS
flags.DEFINE_string("mia_method", "rmia", "MIA method to use. One of ['rmia', 'lira'].")
flags.DEFINE_boolean("log_scale", True, "Whether to plot ROC curves in log scale.")
flags.DEFINE_boolean("plot_std_agg_sucess", False, "whether to plot std (across target models) for ROC curves of aggregate attack sucess.")


def get_color(dataset_name):
    if dataset_name == "chexpert":
        return chex_color
    elif dataset_name == "mimic":
        return mimic_color
    elif dataset_name == "fitzpatrick":
        return fitz_color
    elif dataset_name == "fairvision":
        return fairvision_color
    elif dataset_name == "embed":
        return embed_color
    elif dataset_name == "ptb-xl":
        return ptb_xl_color
    elif dataset_name == "mimic-iv-ed":
        return mimic_iv_ed_color
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}.")



def get_dataset_name(dataset_name):
    if dataset_name == "chexpert":
        return "CheXpert"
    elif dataset_name == "mimic":
        return "MIMIC-CXR"
    elif dataset_name == "fitzpatrick":
        return "Fitzpatrick 17k"
    elif dataset_name == "fairvision":
        return "FairVision"
    elif dataset_name == "embed":
        return "EMBED"
    elif dataset_name == "ptb-xl":
        return "PTB-XL"
    elif dataset_name == "mimic-iv-ed":
        return "MIMIC-IV ED"
    else:
        raise ValueError(f"Unknown dataset name {dataset_name}.")


def main(argv):
    # ROC analysis of average MIA success
    fig, ax = plt.subplots(layout="tight")
    for dataset_name, figdir in FIGDIRS.items():
        files_dir = Path(figdir) / "files"
        scale = "log" if FLAGS.log_scale else "lin"
        aucs = joblib.load(files_dir / f"{FLAGS.mia_method}_aucs.pkl")
        record_level_mia_aucs = np.load(files_dir / "record_aucs.npy")
        if not files_dir.exists():
            raise FileNotFoundError(f"{files_dir} does not exist.")
        # linear scale ROC-curve
        fpr = np.load(files_dir / f"{FLAGS.mia_method}_fpr_{scale}.npy")
        tpr = np.load(files_dir / f"{FLAGS.mia_method}_tpr_mean_{scale}.npy")
        ax.plot(
            fpr,
            np.load(files_dir / f"{FLAGS.mia_method}_tpr_mean_{scale}.npy"),
            label=f"{get_dataset_name(dataset_name)} ({np.mean(record_level_mia_aucs):.2f})",
            color=get_color(dataset_name),
        )
        if not FLAGS.log_scale:
            # plot std of TPR
            ax.fill_between(
                fpr,
                tpr - np.load(files_dir / f"{FLAGS.mia_method}_tpr_std_{scale}.npy"),
                tpr + np.load(files_dir / f"{FLAGS.mia_method}_tpr_std_{scale}.npy"),
                alpha=0.2,
                color=get_color(dataset_name),
            )
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5, label="Random")
    ax.spines[["right", "top"]].set_visible(False)
    if FLAGS.log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([1e-5, 1])
        ax.set_ylim([1e-5, 1])
        ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    else:
        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    #fig.legend(loc="upper right")
    plt.savefig(
        f"./figs/agg_{FLAGS.mia_method}_MIA_ROC_curves_{scale}.pdf", bbox_inches="tight"
    )
    # patient-level ROC analysis (max)
    n_max = 0
    fig, ax = plt.subplots(layout="tight")
    for dataset_name, figdir in FIGDIRS.items():
        files_dir = Path(figdir) / "files"
        if not files_dir.exists():
            raise FileNotFoundError(f"{files_dir} does not exist.")
        aucs = np.load(files_dir / "record_aucs.npy")
        n = len(aucs)
        if n > n_max:
            n_max = n
        try:
            patient_ids = joblib.load(files_dir / "patient_ids.pkl")
        except FileNotFoundError:
            patient_ids = None
        mia_auc_esf_plot(
            aucs,
            patient_ids=patient_ids,
            ax=ax,
            color=get_color(dataset_name),
            label=f"{get_dataset_name(dataset_name)} ({np.mean(aucs):.2f}±{np.std(aucs):.3f})",
        )
    ylim = 1e-5
    ax.set_xlim([0.475, 1])
    ax.set_ylim([ylim, 1])
    ax.set_yscale("log")
    ax.set_xlabel("MIA AUC (Patient-level)")
    ax.set_ylabel("1 - Cumulative Probability")
    #fig.legend(bbox_to_anchor=(0.5, -0.15), loc="lower center", ncol=3)
    plt.savefig(f"./figs/patient_level_MIA_AUC_eSFs.pdf", bbox_inches="tight")
    # patient-level ROC analysis (mean)
    n_max = 0
    fig, ax = plt.subplots(layout="tight")
    for dataset_name, figdir in FIGDIRS.items():
        files_dir = Path(figdir) / "files"
        if not files_dir.exists():
            raise FileNotFoundError(f"{files_dir} does not exist.")
        aucs = np.load(files_dir / "record_aucs.npy")
        n = len(aucs)
        if n > n_max:
            n_max = n
        try:
            patient_ids = joblib.load(files_dir / "patient_ids.pkl")
        except FileNotFoundError:
            patient_ids = None
        mia_auc_esf_plot(
            aucs,
            patient_ids=patient_ids,
            aggregation_mode='mean',
            ax=ax,
            color=get_color(dataset_name),
            label=f"{get_dataset_name(dataset_name)} ({np.mean(aucs):.2f}±{np.std(aucs):.3f})",
        )
    ylim = 1e-5
    ax.set_xlim([0.475, 1])
    ax.set_ylim([ylim, 1])
    ax.set_yscale("log")
    ax.set_xlabel("MIA AUC (Patient-level)")
    ax.set_ylabel("1 - Cumulative Probability")
    # fig.legend(bbox_to_anchor=(0.5, -0.15), loc="lower center", ncol=3)
    plt.savefig(f"./figs/patient_level_MIA_AUC(mean)_eSFs.pdf", bbox_inches="tight")
    # record-level ROC analysis
    fig, ax = plt.subplots(layout="tight")
    for dataset_name, figdir in FIGDIRS.items():
        files_dir = Path(figdir) / "files"
        if not files_dir.exists():
            raise FileNotFoundError(f"{files_dir} does not exist.")
        aucs = np.load(files_dir / "record_aucs.npy")
        n = len(aucs)
        if n > n_max:
            n_max = n
        try:
            patient_ids = joblib.load(files_dir / "patient_ids.pkl")
        except FileNotFoundError:
            patient_ids = None
        mia_auc_esf_plot(
            aucs,
            patient_ids=None,
            ax=ax,
            color=get_color(dataset_name),
            label=f"{get_dataset_name(dataset_name)} ({np.mean(aucs):.2f}±{np.std(aucs):.3f})",
        )
    ylim = 1e-5
    ax.set_xlim([0.475, 1])
    ax.set_ylim([ylim, 1])
    ax.set_yscale("log")
    ax.set_xlabel("MIA AUC (Record-level)")
    ax.set_ylabel("1 - Cumulative Probability")
    # fig.legend(bbox_to_anchor=(0.5, -0.15), loc="lower center", ncol=3)
    plt.savefig(f"./figs/record_level_MIA_AUC_eSFs.pdf", bbox_inches="tight")

    
if __name__ == "__main__":
    app.run(main)
