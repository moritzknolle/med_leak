

def get_dataset_str(dataset: str):
    if dataset == "mimic":
        return "MIMIC-CXR"
    elif dataset == "chexpert":
        return "CheXpert"
    elif dataset == "fitzpatrick":
        return "Fitzpatrick-17k"
    elif dataset == "ukbb_cfp":
        return "UKBB-CFP"
    elif dataset == "embed":
        return "EMBED"
    elif dataset == "fairvision":
        return "FairVision"
    raise ValueError(f"Invalid dataset name., {dataset}")
