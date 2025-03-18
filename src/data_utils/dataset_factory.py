from pathlib import Path

import numpy as np
import pandas as pd

from src.data_utils.datasets import (CXRDataset, EMBEDataset,
                                           FairVisionDataset,
                                           FitzPatLabelSetting,
                                           FitzpatrickDataset)


def get_dataset(
    dataset_name: str,
    img_size: int,
    csv_root: Path,
    data_root: Path,
    save_root: Path,
    one_image_per_patient: bool = False,
    get_numpy: bool = False,
    load_from_disk: bool = False,
    overwrite_existing: bool = False,
    frontal_only: bool = False,
):
    """
    Convenience function to retrieve a dataset by name. Raises a ValueError if the dataset is unknown.
        Args:
            dataset_name: str, name of the dataset
            img_size: int, size of the images
            csv_root: Path, path to the csv files
            data_root: Path, path to the data files
            save_root: Path, path to the save root
            one_image_per_patient: bool, whether to use only one image per patient
            get_numpy: bool, whether to return numpy arrays
            load_from_disk: bool, whether to load from disk
            overwrite_existing: bool, whether to overwrite existing files
            frontal_only: bool, whether to use only frontal images (only for MIMIC-CXR/Chexpert)
        Returns:
            train_dataset: either tuple of numpy arrays or a BaseDataset object
    """
    if dataset_name == "chexpert":
        cxp_df_path = (
            csv_root / "chexpert_train.csv"
            if one_image_per_patient
            else csv_root / "chexpert_train_full.csv"
        )
        cxp_df = pd.read_csv(cxp_df_path)
        cxp_test_df = pd.read_csv(csv_root / "chexpert_test.csv")
        use_memmap = img_size > 128
        train_dataset = CXRDataset(
            df=cxp_df,
            img_path=data_root,
            name="chexpert" if one_image_per_patient else "chexpert_full",
            img_size=img_size,
            split="train",
            save_root=save_root,
            memmap=use_memmap,
            frontal_only=frontal_only,
        )
        test_dataset = CXRDataset(
            df=cxp_test_df,
            img_path=data_root,
            name="chexpert" if one_image_per_patient else "chexpert_full",
            img_size=img_size,
            split="test",
            save_root=save_root,
        )
    elif dataset_name == "mimic":
        csv_path = (
            csv_root / "mimic_df.csv"
            if one_image_per_patient
            else csv_root / "mimic_df_full.csv"
        )
        mimic_df = pd.read_csv(csv_path)
        mimic_train = mimic_df[mimic_df.split == "train"]
        use_memmap = img_size > 128
        train_dataset = CXRDataset(
            df=mimic_train,
            img_path=data_root / "files",
            name="mimic" if one_image_per_patient else "mimic_full",
            img_size=img_size,
            split="train",
            save_root=save_root,
            memmap=use_memmap,
            frontal_only=frontal_only,
        )
        # use CheXpert test set since labels are verified by experts (consensus voting of three board-certified radiologists)
        cxp_test_df = pd.read_csv(csv_root / "chexpert_test.csv")
        test_dataset = CXRDataset(
            df=cxp_test_df,
            img_path=data_root,
            name="chexpert" if one_image_per_patient else "chexpert_full",
            img_size=img_size,
            split="test",
            save_root=save_root,
        )
    elif dataset_name == "fitzpatrick":
        fitz_df = pd.read_csv(f"{csv_root}/fitzpatrick17k.csv")
        fitz_df = fitz_df[fitz_df["download_success"] == True]
        fitz_train_df = fitz_df[fitz_df["split"] == "train"]
        fitz_test_df = fitz_df[fitz_df["split"] == "test"]
        train_dataset = FitzpatrickDataset(
            df=fitz_train_df,
            img_path=data_root,
            name="fitzpatrick",
            img_size=img_size,
            split="train",
            save_root=save_root,
            label_setting=FitzPatLabelSetting.COARSE,
        )
        test_dataset = FitzpatrickDataset(
            df=fitz_test_df,
            img_path=data_root,
            name="fitzpatrick",
            img_size=img_size,
            split="test",
            save_root=save_root,
            label_setting=FitzPatLabelSetting.COARSE,
        )
    elif dataset_name == "embed":
        embed_df = (
            pd.read_csv("./data/csv/embed.csv")
            if one_image_per_patient
            else pd.read_csv("./data/csv/embed_full.csv")
        )
        train_df, test_df = (
            embed_df[embed_df.split == "train"],
            embed_df[embed_df.split == "test"],
        )
        use_memmap = img_size > 128
        train_dataset = EMBEDataset(
            df=train_df,
            img_path=data_root,
            name="embed" if one_image_per_patient else "embed_full",
            img_size=img_size,
            split="train",
            save_root=save_root,
            memmap=use_memmap,
        )
        test_dataset = EMBEDataset(
            df=test_df,
            img_path=data_root,
            name="embed" if one_image_per_patient else "embed_full",
            img_size=img_size,
            split="test",
            save_root=save_root,
            memmap=use_memmap,
        )
    elif dataset_name == "fairvision":
        fairvision_df = pd.read_csv(csv_root / "fairvision.csv")
        train_df, test_df = (
            fairvision_df[fairvision_df.use != "test"],
            fairvision_df[fairvision_df.use == "test"],
        )
        train_dataset = FairVisionDataset(
            df=train_df,
            img_path=data_root,
            name="fairvision",
            img_size=img_size,
            split="train",
            save_root=save_root,
        )
        test_dataset = FairVisionDataset(
            df=test_df,
            img_path=data_root,
            name="fairvision",
            img_size=img_size,
            split="test",
            save_root=save_root,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    if get_numpy:
        # extract np.arrays
        print("... building numpy arrays")
        x_train, y_train = train_dataset.get_numpy(
            load_from_disk=load_from_disk, overwrite_existing=overwrite_existing
        )
        x_test, y_test = test_dataset.get_numpy(
            load_from_disk=load_from_disk, overwrite_existing=overwrite_existing
        )
        # add channel dimension if necessary
        if len(x_train.shape) == 3:
            x_train = np.expand_dims(x_train, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)
        # convert to channels last if necessary
        if not x_train.shape[-1] in [1, 3]:
            x_train = np.transpose(x_train, (0, 2, 3, 1))
            x_test = np.transpose(x_test, (0, 2, 3, 1))
        # shape checks
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
        assert x_train.shape[1:] == x_test.shape[1:]
        print("... done building numpy arrays")
        return (x_train, y_train), (x_test, y_test)
    return train_dataset, test_dataset
