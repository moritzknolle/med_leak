from functools import partial
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from src.data_utils.dataset_factory import get_dataset
from src.data_utils.utils import get_dataset_str
from src.privacy_utils.lira import (compute_scores,
                                          loss_logit_transform_multiclass,
                                          perform_lira,
                                          record_MIA_ROC_analysis)
from src.privacy_utils.common import load_score
from src.privacy_utils.plot_utils import mia_auc_esf_plot, subgroup_plots, disease_group_plots, plot_images
from src.privacy_utils.rmia import (perform_rmia,
                                          rmia_transform_multiclass)
from src.privacy_utils.utils import confidence_roc_plot
from src.train_utils.utils import get_label_mode

FONT_SIZE = 7
plt.style.use("default")
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
        # "pdf.fonttype":42,
    }
)
color_a = "#7c8483"
color_b = "#982649"

flags.DEFINE_string("logdir", "./logs/fitzpatrick/", "The log directory.")
flags.DEFINE_string(
    "dataset",
    "fitzpatrick",
    "The dataset to analyse. This script will try to look for log directories at FLAGS.logdir",
)
flags.DEFINE_string(
    "csv_root", "./data/csv/", "The directory where the csv files can be found."
)
flags.DEFINE_string(
    "save_root",
    "/home/moritz/data_fast/npy",
    "The directory where the pre-computed .npy dataset files can be found.",
)
flags.DEFINE_string(
    "label_mode",
    "multiclass",
    "The label mode to use. One of ['binary', 'multiclass', 'multilabel', 'simclr']",
)
flags.DEFINE_integer("img_size", 128, "Image size (for plotting only).")
flags.DEFINE_integer("n_models", 200, "Total number of models to use for the attack.")
flags.DEFINE_integer(
    "eval_size", 10, "Number of samples/training runs to use for validation."
)
flags.DEFINE_bool(
    "shuffle_dirs",
    True,
    "Whether to shuffle the log directories before computing the scores.",
)
flags.DEFINE_bool(
    "patient_level_only",
    False,
    "If set to True, perform patient-level MIA performance analysis only.",
)
flags.DEFINE_bool(
    "plot_images",
    True,
    "If set to True, plot some sample images most vulnerable records and randomly sample records.",
)
flags.DEFINE_bool(
    "offline",
    False,
    "Whether to use the offline version of the implemented attacks. This requires only models not trained on the target record",
)
flags.DEFINE_float(
    "offline_a",
    1.0,
    "Offline approximation parameter a used for RMIA. See original paper for details.",
)
flags.DEFINE_bool(
    "find_optimal_a",
    False,
    "Whether to perform a grid search to find the optimal value of a.",
)
flags.DEFINE_bool(
    "fix_variance", False, "Whether to estimate a global variance for all records."
)
flags.DEFINE_float(
    "ylim_tpr",
    1e-5,
    "Minimum true positive rate (to set ylim correctly and consistently) for log-log ROC curve plots.",
)
flags.DEFINE_boolean(
    "multiprocessing", True, "Whether to use multiprocessing for parallel computation."
)
flags.DEFINE_integer("threads", 16, "Number of threads to use for multiprocessing.")
flags.DEFINE_integer("seed", 21, "Random seed.")
FLAGS = flags.FLAGS


def fig_dir_exists(out_dir: Path):
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    files_dir = out_dir / "files"
    if not files_dir.exists():
        files_dir.mkdir(parents=True, exist_ok=True)


def get_patient_col(dataset_name: str):
    if dataset_name == "chexpert" or dataset_name == "fairvision":
        patient_col = "patient_id"
    elif dataset_name == "mimic":
        patient_col = "subject_id"
    elif dataset_name == "embed":
        patient_col = "empi_anon"
    elif dataset_name == "fitzpatrick":
        raise ValueError("Fitzpatrick does not have patient IDs.")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return patient_col

def get_data_root(dataset_name: str):
    if dataset_name == "chexpert":
        data_root=Path("/home/moritz/data/chexpert")
    elif dataset_name == "mimic":
        data_root=Path("/home/moritz/data/mimic-cxr/mimic-cxr-jpg")
    elif dataset_name == "fairvision":
        data_root=Path("/home/moritz/data_big/fairvision/FairVision")
    elif dataset_name == "embed":
        data_root=Path("/home/moritz/data_huge/embed_small/processed_512x512")
    elif dataset_name == "fitzpatrick":
        data_root=Path("/home/moritz/data/fitzpatrick17k")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return data_root

def main(argv):
    np.random.seed(FLAGS.seed)
    print("... loading MIA results for dataset", FLAGS.dataset, "from", FLAGS.logdir)
    train_dataset, _ = get_dataset(
        dataset_name=FLAGS.dataset,
        img_size=FLAGS.img_size,
        csv_root=Path(FLAGS.csv_root),
        save_root=Path(FLAGS.save_root),
        data_root=get_data_root(FLAGS.dataset),
    )
    base_dir = Path(FLAGS.logdir)
    out_dir = Path(f"./figs/{FLAGS.dataset}")
    fig_dir_exists(out_dir)
    logdirs = [d for d in base_dir.iterdir()]
    label_mode = get_label_mode(FLAGS.label_mode)
    is_mimic_or_chex = FLAGS.dataset in ["mimic", "chexpert"]

    logit_transform_func = partial(
        loss_logit_transform_multiclass, is_mimic_or_chexpert=is_mimic_or_chex
    )
    rmia_transform_func = partial(
        rmia_transform_multiclass, is_mimic_or_chexpert=is_mimic_or_chex
    )
    logit_load_func = partial(load_score, logit_transform_func=logit_transform_func)
    rmia_load_func = partial(load_score, logit_transform_func=rmia_transform_func)

    if FLAGS.shuffle_dirs:
        np.random.shuffle(logdirs)
    if FLAGS.n_models is not None:
        logdirs = logdirs[: FLAGS.n_models]

    scores, masks, labels = compute_scores(
        log_dirs=logdirs,
        load_score_func=logit_load_func,
        multi_processing=FLAGS.multiprocessing,
        threads=FLAGS.threads,
    )
    if not FLAGS.patient_level_only:
        rmia_scores, rmia_masks, _ = compute_scores(
            log_dirs=logdirs,
            load_score_func=rmia_load_func,
            multi_processing=FLAGS.multiprocessing,
            threads=FLAGS.threads,
        )
        print("RMIA shapes:", rmia_scores.shape, rmia_masks.shape)
        assert np.allclose(masks, rmia_masks), "Careful, masks do not match."
    assert scores.shape[1] == len(
        train_dataset
    ), f"Shape mismatch: {scores.shape} vs {len(train_dataset)}"
    assert masks.shape[1] == len(
        train_dataset
    ), f"Shape mismatch: {masks.shape} vs {len(train_dataset)}"
    print(
        f"scores: shape={scores.shape} mean={scores.mean():.3f} max={scores.max():.3f} min={scores.min():.3f}, masks: shape={masks.shape}"
    )
    # patient-level analytic (record-level) ROC anaylsis using all scores
    _, _, record_aucs, record_aucs_se = record_MIA_ROC_analysis(
        scores=scores, masks=masks, resolution=10_000, log_scale=False
    )
    np.save(out_dir / "files" / "record_aucs.npy", record_aucs)
    # patient-level MIA eSF plot
    if FLAGS.dataset != "fitzpatrick":
        patient_id_col = get_patient_col(FLAGS.dataset)
        assert (
            patient_id_col in train_dataset.dataframe.columns
        ), f"Patient ID column {patient_id_col} not found in dataframe. \n{train_dataset.dataframe.columns}"
        patient_ids = train_dataset.dataframe[patient_id_col]
        mia_auc_esf_plot(
            record_aucs,
            patient_ids=patient_ids,
            save_path=f"./figs/{FLAGS.dataset}/{FLAGS.dataset}_patient_level_auc_eSF.pdf",
        )
        joblib.dump(patient_ids, out_dir / "files" / "patient_ids.pkl")
    # record-level MIA eSF plot
    mia_auc_esf_plot(
        record_aucs,
        save_path=f"./figs/{FLAGS.dataset}/{FLAGS.dataset}_record_level_auc_eSF.pdf",
    )
    if FLAGS.plot_images:
        # plot most vulnerable records
        plot_images(risk_scores=record_aucs, dataset=train_dataset, dataset_name=FLAGS.dataset, out_dir=out_dir, highest=True)
        plot_images(risk_scores=record_aucs, dataset=train_dataset, dataset_name=FLAGS.dataset, out_dir=out_dir, highest=False)

    if not FLAGS.patient_level_only:
        train_scores, test_scores = (
            scores[: -FLAGS.eval_size],
            scores[-FLAGS.eval_size :],
        )
        train_rmia_scores, test_rmia_scores = (
            rmia_scores[: -FLAGS.eval_size],
            rmia_scores[-FLAGS.eval_size :],
        )
        train_masks, test_masks = masks[: -FLAGS.eval_size], masks[-FLAGS.eval_size :]
        lira_preds, lira_targets = perform_lira(
            train_scores=train_scores,
            train_masks=train_masks,
            test_scores=test_scores,
            test_masks=test_masks,
            fix_variance=FLAGS.fix_variance,
            offline=FLAGS.offline,
            label_type=label_mode,
            multi_processing=FLAGS.multiprocessing,
            threads=FLAGS.threads,
        )

        rmia_preds, rmia_targets = perform_rmia(
            train_scores=train_rmia_scores,
            train_masks=train_masks,
            test_scores=test_rmia_scores,
            test_masks=test_masks,
            find_optimal_a=FLAGS.find_optimal_a,
            offline=FLAGS.offline,
            a=FLAGS.offline_a,
            label_type=label_mode,
            labels=labels if FLAGS.label_mode == "multilabel" else None,
            multi_processing=FLAGS.multiprocessing,
            threads=FLAGS.threads,
        )
        # privacy analysis by aggretate atttack sucess ROC Curve
        lira_aucs = [roc_auc_score(t, p) for t, p in zip(lira_targets, lira_preds)]
        rmia_aucs = [roc_auc_score(t, p) for t, p in zip(rmia_targets, rmia_preds)]
        fig, axes = plt.subplots(
            1, 2, figsize=(4, 2), sharey=False, sharex=False, layout="constrained"
        )
        joblib.dump(lira_aucs, out_dir / "files" / "lira_aucs.pkl")
        joblib.dump(rmia_aucs, out_dir / "files" / "rmia_aucs.pkl")
        lim_fpr = 1e-5
        confidence_roc_plot(
            preds=lira_preds,
            targets=lira_targets,
            color=color_a,
            log_scale=False,
            name=f"LiRA:({np.mean(lira_aucs):.3f}±{np.std(lira_aucs):.3f})",
            draw_legend=False,
            draw_axes_labels=False,
            draw_random=False,
            ax=axes[0],
            save_result=True,
            out_path=out_dir / "files",
            prefix_str="lira",
            lim_fpr=lim_fpr,
        )
        confidence_roc_plot(
            preds=rmia_preds,
            targets=rmia_targets,
            color=color_b,
            log_scale=False,
            name=f"RMIA({np.mean(rmia_aucs):.3f}±{np.std(rmia_aucs):.3f})",
            draw_legend=False,
            draw_axes_labels=False,
            ax=axes[0],
            save_result=True,
            out_path=out_dir / "files",
            prefix_str="rmia",
            lim_fpr=lim_fpr,
        )
        confidence_roc_plot(
            preds=lira_preds,
            targets=lira_targets,
            color=color_a,
            log_scale=True,
            name=f"LiRA:({np.mean(lira_aucs):.3f}±{np.std(lira_aucs):.3f})",
            lim_tpr=FLAGS.ylim_tpr,
            draw_axes_labels=False,
            draw_legend=False,
            draw_random=False,
            ax=axes[1],
            save_result=True,
            out_path=out_dir / "files",
            prefix_str="lira",
            lim_fpr=lim_fpr,
        )
        confidence_roc_plot(
            preds=rmia_preds,
            targets=rmia_targets,
            color=color_b,
            log_scale=True,
            name=f"RMIA({np.mean(rmia_aucs):.3f}±{np.std(rmia_aucs):.3f})",
            lim_tpr=FLAGS.ylim_tpr,
            draw_axes_labels=False,
            draw_legend=True,
            ax=axes[1],
            save_result=True,
            out_path=out_dir / "files",
            prefix_str="rmia",
            lim_fpr=lim_fpr,
        )
        plt.subplots_adjust(wspace=0.1)
        dataset_str = get_dataset_str(FLAGS.dataset)
        fig.suptitle(f"{dataset_str}")
        fig.supxlabel("False Positive Rate")
        fig.supylabel("True Positive Rate")
        plt.savefig(
            f"./figs/{FLAGS.dataset}/{FLAGS.dataset}_MIA_roc_curves.pdf",
            bbox_inches="tight",
        )
        plt.show()
        plt.close()
    # sub-group analysis
    df = train_dataset.dataframe.copy()
    df["MIA_AUC"] = record_aucs
    print(len(df), df.head())
    print(type(df))
    subgroup_plots(dataset_name=FLAGS.dataset, results_df=df, out_path=out_dir)
    print(df.head())
    print(df.columns)
    disease_group_plots(dataset_name=FLAGS.dataset, results_df=df, out_path=out_dir)



if __name__ == "__main__":
    app.run(main)
