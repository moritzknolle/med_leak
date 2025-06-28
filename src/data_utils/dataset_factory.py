from pathlib import Path

import numpy as np
import pandas as pd

from src.data_utils.datasets import (CXRDataset, EMBEDataset,
                                           FairVisionDataset,
                                           FitzPatLabelSetting,
                                           FitzpatrickDataset,
                                           PTBXLDataset,
                                           MIMICIVEDDataset)


def get_dataset(
    dataset_name: str,
    img_size: int,
    csv_root: Path,
    data_root: Path,
    save_root: Path,
    get_numpy: bool = False,
    load_from_disk: bool = False,
    overwrite_existing: bool = False,
):
    """
    Convenience function to retrieve a dataset by name. Raises a ValueError if the dataset is unknown.
        Args:
            dataset_name: str, name of the dataset
            img_size: int, size of the images
            csv_root: Path, path to the csv files
            data_root: Path, path to the data files
            save_root: Path, path to the save root
            get_numpy: bool, whether to return numpy arrays
            load_from_disk: bool, whether to load from disk
            overwrite_existing: bool, whether to overwrite existing files
        Returns:
            train_dataset: either tuple of numpy arrays or a BaseDataset object
    """
    if dataset_name == "chexpert":
        cxp_df_path = (
            csv_root / "chexpert_train.csv"
        )
        cxp_df = pd.read_csv(cxp_df_path)
        cxp_test_df = pd.read_csv(csv_root / "chexpert_test.csv")
        use_memmap = img_size > 128
        train_dataset = CXRDataset(
            df=cxp_df,
            img_path=data_root,
            name="chexpert",
            img_size=img_size,
            split="train",
            save_root=save_root,
            memmap=use_memmap,
        )
        test_dataset = CXRDataset(
            df=cxp_test_df,
            img_path=data_root,
            name="chexpert",
            img_size=img_size,
            split="test",
            save_root=save_root,
        )
    elif dataset_name == "mimic":
        csv_path = (
            csv_root / "mimic_train.csv"
        )
        mimic_df = pd.read_csv(csv_path)
        use_memmap = img_size > 128
        train_dataset = CXRDataset(
            df=mimic_df,
            img_path=data_root / "files",
            name="mimic",
            img_size=img_size,
            split="train",
            save_root=save_root,
            memmap=use_memmap,
        )
        # use CheXpert test set since labels are verified by experts (consensus voting of three board-certified radiologists)
        cxp_test_df = pd.read_csv(csv_root / "chexpert_test.csv")
        test_dataset = CXRDataset(
            df=cxp_test_df,
            img_path=data_root,
            name="chexpert",
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
        )
        train_df, test_df = (
            embed_df[embed_df.split == "train"],
            embed_df[embed_df.split == "test"],
        )
        use_memmap = img_size > 128
        train_dataset = EMBEDataset(
            df=train_df,
            img_path=data_root,
            name="embed",
            img_size=img_size,
            split="train",
            save_root=save_root,
            memmap=use_memmap,
        )
        test_dataset = EMBEDataset(
            df=test_df,
            img_path=data_root,
            name="embed",
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
    elif dataset_name == "ptb-xl":
        train_df = pd.read_csv(csv_root / "ptb-xl_train.csv")
        test_df = pd.read_csv(csv_root / "ptb-xl_test.csv")
        train_dataset = PTBXLDataset(
            df=train_df,
            img_path=data_root,
            name="ptb-xl",
            split="train",
            save_root=save_root,
        )
        test_dataset = PTBXLDataset(
            df=test_df,
            img_path=data_root,
            name="ptb-xl",
            split="test",
            save_root=save_root,
        )
    elif dataset_name == "mimic-iv-ed":
        train_df = pd.read_csv(csv_root / "mimic-iv-ed_train.csv")
        test_df = pd.read_csv(csv_root / "mimic-iv-ed_test.csv")
        train_dataset = MIMICIVEDDataset(
            dataframe=train_df
        )
        test_dataset = MIMICIVEDDataset(
            dataframe=test_df
        )
        # MIMICIVEDDataset does not support iterative loading of data, so we return the full data directly
        if get_numpy:
            x_train, y_train = train_dataset.__get_all_inputs__(), train_dataset.__get_all_targets__()
            x_test, y_test = test_dataset.__get_all_inputs__(), test_dataset.__get_all_targets__()
            return (x_train, y_train), (x_test, y_test)
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
        if len(x_train.shape) == 3 and x_train.shape[1]==x_train.shape[2]:
            x_train = np.expand_dims(x_train, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)
        # convert to channels last if necessary
        if not x_train.shape[-1] in [1, 3, 12]:
            x_train = np.transpose(x_train, (0, 2, 3, 1))
            x_test = np.transpose(x_test, (0, 2, 3, 1))
        # shape checks
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
        assert x_train.shape[1:] == x_test.shape[1:]
        print("... done building numpy arrays")
        return (x_train, y_train), (x_test, y_test)
    return train_dataset, test_dataset
