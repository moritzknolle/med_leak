from pathlib import Path

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
        patient_col = "md5hash" # we assume each image belongs to a unique patient
    elif dataset_name == "ptb-xl":
        patient_col = "patient_id"
    elif dataset_name == "mimic-iv-ed":
        patient_col = "subject_id"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return patient_col


def get_data_root(dataset_name: str):
    if dataset_name == "chexpert":
        data_root = Path("/home/moritz/data/chexpert")
    elif dataset_name == "mimic":
        data_root = Path("/home/moritz/data/mimic-cxr/mimic-cxr-jpg")
    elif dataset_name == "fairvision":
        data_root = Path("/home/moritz/data_big/fairvision/FairVision")
    elif dataset_name == "embed":
        data_root = Path("/home/moritz/data_massive/embed_small/png/1024x768")
    elif dataset_name == "fitzpatrick":
        data_root = Path("/home/moritz/data/fitzpatrick17k")
    elif dataset_name == "ptb-xl":
        data_root = Path("/home/moritz/data/physionet.org/files/ptb-xl/1.0.3/")
    elif dataset_name == "mimic-iv-ed":
        data_root = None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return data_root