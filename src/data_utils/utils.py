

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
    elif dataset == "ptb-xl":
        return "PTB-XL"
    elif dataset == "mimic-iv-ed":
        return "MIMIC-IV-ED"
    raise ValueError(f"Invalid dataset name., {dataset}")
