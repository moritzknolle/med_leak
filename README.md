
# MedLeak: patient-level privacy audits of medical AI

Code to reproduce results from our paper **"Disparate privacy risks from medical AI"**.


## Installation

This codebase requires a functional [jax](https://docs.jax.dev) installation with GPU support. Up-to-date installation instructions can be found [here](https://docs.jax.dev/en/latest/installation.html#conda-installation).

```bash
    conda create -n med-leak python=3.9.18
    conda activate med-leak
    # install jax now by following the instructions at https://docs.jax.dev/en/latest/installation.html
    git clone https://github.com/moritzknolle/med_leak.git
    cd med_leak
    pip install -r requirements.txt
```
    
## Datasets
All investigated datasets are publicly available for research purposes, see table below for access links.

| **Name** | **# Patients** | **# Images** |
|------------|--------------|------------|
| [CheXpert](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf) | 65,420 | 224,316 |
| [CheXpert (demographic)](https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf) | 65,420 | 224,316 |
| [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) | 65,379 | 377,110 |
| [MIMIC-CXR (demographic)](https://physionet.org/content/mimiciv/1.0/) | 65,379 | 377,110 |
| [Fitzpatrick-17k](https://github.com/mattgroh/fitzpatrick17k) | n.a. | 16,523 |
| [Harvard FairVision](https://ophai.hms.harvard.edu/datasets/harvard-fairvision30k) | 30,000 | 30,000 |
| [EMBED](https://registry.opendata.aws/emory-breast-imaging-dataset-embed/) | 23,057 | 451,642 |

Dataset-specific pre-processing code can be found in ```src/data_utils/notebooks/```.
## Example: Reproducing patient-level audit results for Fitzpatrick 17k
Patient/record-level privacy audits require measuring attack performance individually for each record across many target models.
Training many target models can be computationally expensive, thus we recommend to start out with *Fitzpatrick 17k*. For this dataset training a highly performant WRN-28-2 model only takes 2 mins (on a single A100 GPU). The commands below will train 50 target models and then compute and visualise aggregate, record- and patient-level attack success.   

```bash
conda activate med-leak
python fitzpatrick.py --n_runs=50 --eval_only=False --logdir="./logs/fitzpatrick" # train reference models
python mem_inf_stats.py --logdir="./logs/fitzpatrick" --dataset='fitzpatrick" # perform attacks and plot results
```


## Reproducing attacks on open-source models
Requires a additional [PyTorch](https://pytorch.org/) and [TorchXrayVision](https://mlmed.org/torchxrayvision/) installations (see [here](https://pytorch.org/get-started/locally/) for up-to-date installation instructions). To replicate results for attacking the CheXpert and MIMIC-CXR models from TorchXrayVision, run the following commands below.

```bash
conda activate torch # activate your environment with Pytorch and TorchXrayVision installed
python xrv_scores.py # model inference
conda activate med-leak
python xrv_attack.py # conduct attacks
```

## Reproducing all results

To reproduce all quantitative results presented in the paper simply run the commands below.

```bash
bash scripts/train_all.sh
bash scripts/plot_all.sh
```

**WARNING**: this will take a LONG time (~800h on single A100 GPU)! Fortunately, you can accelerate the process with more GPUs. Simply (re)-run each line in ```scripts/train_all.sh``` as many times as you like for each GPU-machine you have available (underlying logic in ```src/train_utils/training.py``` will handle concurrency).