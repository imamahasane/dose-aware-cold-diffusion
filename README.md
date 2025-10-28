# Dose-Aware Cold Diffusion with Physics Consistency for Generalizable Low-Dose CT Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official implementation of the paper:

> **“Dose-Aware Cold Diffusion with Physics Consistency for Generalizable Low-Dose CT Reconstruction”**  
> *Md Imam Ahasan, Guangchao Yang, A. F. M. Abdun Noor, Mohammad Azam Khan, Jaegul Choo*  
> Submitted to **IEEE Transactions on Medical Imaging**, October 2025.

---

## Overview

Low-dose CT (LDCT) reconstruction aims to reduce patient radiation exposure while maintaining diagnostic image quality.  
Existing deep-learning and diffusion-based methods often fail to generalize across unseen dose levels.

**Dose-Aware Cold Diffusion (DACD)** introduces a *physics-consistent*, *dose-conditioned* diffusion framework that integrates:

- **Dose-Aware Perception (DAP):** continuous embedding of dose level \(d \in (0, 1]\)  
- **PEM⁺⁺ Module:** multi-scale prior extraction (gradient, Laplacian, non-local means, and CNN features)  
- **Dose-Calibrated Step Allocation (DCSA):** adaptive diffusion scheduling per dose severity  
- **Physics Consistency Loop:** forward–backprojection correction within each denoising step  
- **Cold Diffusion Formulation:** physically simulated Poisson thinning instead of Gaussian noise

The framework generalizes across *seen and unseen* dose levels on **Mayo-2020**, **Mayo-2016**, and **LoDoPaB-CT** datasets.

---

## Key Features

- **Continuous dose-aware conditioning** for robust denoising  
- **Physics-consistent correction** enforcing projection-domain fidelity  
- **FBP warm-start initialization** for faster, stable convergence  
- **Two-stage training** (perception → reconstruction)  
- **Extensive evaluation & statistics** (PSNR, SSIM, RMSE; *optional*: NPS/CNR; Wilcoxon + BH-FDR)

---

## Installation

```bash
git clone https://github.com/imamahasane/physics-guided-cold-diffusion.git
cd physics-guided-cold-diffusion
pip install -r requirements.txt
```


## Dataset Preparation

1. **Download datasets**
   - [Mayo Clinic 2020 Low-Dose CT Grand Challenge](https://www.aapm.org/GrandChallenge/LowDoseCT/)
   - Optionally, also prepare:
     - **Mayo-2016 LDCT dataset**
     - **LoDoPaB-CT** dataset from [https://lodopab.in.tum.de/](https://lodopab.in.tum.de/)

2. **Preprocess data**
   The preprocessing step converts DICOMs to Hounsfield Units (HU), clips intensity to the clinically relevant range \([-1024, 3072]\), and simulates low-dose sinograms using Poisson thinning.

   ```bash
   python scripts/preprocess_data.py \
       --dataset mayo2020 \
       --input_dir /path/to/mayo/raw \
       --output_dir /path/to/processed/mayo2020 \
       --dose_levels 0.50 0.25 0.125 0.05 \
       --fbp_filter hann \
       --geometry parallel \
       --n_angles 720 \
       --detector_bins 736
   ```


## Training
Stage A: Train Dose-Aware Perception Module
```bash
python scripts/train_stage_a.py \
    --config configs/mayo2020_stageA.yaml \
    --data_path /path/to/processed/mayo2020/h5/train_000.h5

```
Stage B: Train Reconstruction Diffusion with Physics Consistency
```bash
python scripts/train_stage_b.py \
    --config configs/mayo2020_stageB.yaml \
    --data_path /path/to/processed/mayo2020/h5/train_000.h5
```
## Evaluation
Run inference and quantitative evaluation on held-out test data:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/dacd_stageB_best.pth \
    --data_path /path/to/processed/mayo2020/h5/test_000.h5 \
    --results_dir results/mayo2020
```

## Project Structure
```bash
physics-guided-cold-diffusion/
├── configs/           # YAML configuration files
├── data/              # Data loading & preprocessing utilities
├── models/            # DAP, PEM++, DCSA, and UNet-based architectures
├── ops/               # Physics operators (A, Aᵀ, FBP) and Poisson dose simulation
├── train/             # Training routines and losses
├── eval/              # Evaluation metrics and statistical analysis
├── scripts/           # Utility scripts for dataset prep and experiments
├── results/           # Quantitative tables and visual outputs
└── tests/             # Unit and smoke tests for reproducibility
```

## Contact
1. Md Imam Ahasan - emamahasane@gmail.com
2. A. F. M Abdun Noor - abdunnoor11@gmail.com
