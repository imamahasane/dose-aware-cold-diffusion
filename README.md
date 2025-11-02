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
LDCT reconstruction aims to minimize patient radiation exposure while preserving diagnostic fidelity.
However, most existing deep-learning and diffusion-based reconstruction approaches struggle to generalize across unseen dose levels due to their fixed noise assumptions and lack of physical constraints.
**Dose-Aware Cold Diffusion (DACD)** introduces a physics-consistent, dose-conditioned diffusion framework that unifies data-driven learning with CT imaging physics.

**Core Innovations:**
- **Dose-Aware Perception (DAP) :** Learns continuous dose embeddings ( d ∈ (0, 1] ) and severity ranking to condition the diffusion model.
- **PEM⁺⁺ (Prior Extraction Module) :** Extracts and fuses structural priors: gradient, Laplacian, non-local means, and CNN-based features.
- **Dose-Calibrated Step Allocation (DCSA) :** Adapts diffusion steps per dose severity for balanced convergence.
- **Physics Consistency Loop (PC) :** Enforces projection-domain fidelity via differentiable forward–backprojection correction.
- **Cold Diffusion Formulation :** Uses Poisson thinning (dose-dependent degradation) instead of Gaussian noise for realism.

DACD generalizes across both seen and unseen dose levels on **Mayo-2020**, **Mayo-2016**, and **LoDoPaB-CT** datasets.

---

## Key Features

- Continuous **dose-aware conditioning** for unseen dose generalization
- **Physics-consistent correction** ensuring measurement fidelity
- **FBP warm-start** for faster, stable training
- **Two-stage training pipeline** (Perception → Reconstruction)
- Comprehensive evaluation: PSNR, SSIM, RMSE, optional NPS/CNR
- Robust statistical testing (Wilcoxon, Cliff’s δ, BH-FDR)

---

## Installation

```bash
git clone https://github.com/imamahasane/physics-guided-cold-diffusion.git
cd physics-guided-cold-diffusion
pip install -r requirements.txt
```
**Optional (recommended for reproducibility):**
```bash
conda env create -f env.yml
conda activate dacd
```

---

## Dataset Preparation

**Download Public Datasets**
- Mayo Clinic 2020 Low-Dose CT Challenge
- Optional:
  - Mayo-2016 LDCT Dataset
  - LoDoPaB-CT Dataset

**Preprocess Data**
Converts DICOMs → Hounsfield Units (HU), clips to clinical range ([-1024, 3072]), and simulates dose-dependent sinograms via Poisson thinning.
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

---

## Training Pipeline
**Stage A — Dose-Aware Perception (DAP)**
Learns continuous dose embeddings and ranking.
```bash
python scripts/train_stage_a.py \
  --config configs/mayo2020_stageA.yaml \
  --data_path /path/to/processed/mayo2020/h5/train_000.h5
```
**Stage B — Reconstruction Diffusion with Physics Consistency**
Trains DACD with physics correction in the reverse diffusion step.
```bash
python scripts/train_stage_b.py \
  --config configs/mayo2020_stageB.yaml \
  --data_path /path/to/processed/mayo2020/h5/train_000.h5
```
Training details:
- Optimizer: **AdamW (lr=1e-4, weight_decay=1e-2)**
- Scheduler: **Cosine annealing with warm-up**
- Mixed precision (AMP)
- Gradient accumulation & checkpointing

---

## Evaluation
Run inference and compute quantitative metrics:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/dacd_stageB_best.pth \
    --data_path /path/to/processed/mayo2020/h5/test_000.h5 \
    --results_dir results/mayo2020
```

**Metrics:** PSNR, SSIM, RMSE, and optional NPS/CNR
**Statistics:** Paired Wilcoxon test, Cliff’s δ, Benjamini–Hochberg FDR correction.

---

## Project Structure
```bash
dose-aware-cold-diffusion/
├── configs/ # YAML configs for datasets, model, training
├── data/ # Data loading, Poisson dose simulation, FBP ops
├── models/ # DAP, PEM++, DCSA, and UNet-based backbones
├── ops/ # Differentiable CT physics (A, Aᵀ, FBP)
├── train/ # Training engines and loss functions
├── eval/ # Metrics, stats, and visualization scripts
├── scripts/ # CLI tools for data prep, train, eval, ablation
├── results/ # Tables and reconstructed images
└── tests/ # Unit, regression, and smoke tests
```
---

## Reproducibility Notes

- Fixed seeds (torch, numpy, random)
- Deterministic cuDNN + AMP for mixed precision
- Implemented with TorchRadon for differentiable CT physics
- Verified on NVIDIA RTX 4090 (CUDA 12.1)

---

## Contact

1. Md Imam Ahasan - emamahasane@gmail.com
2. A. F. M Abdun Noor - abdunnoor11@gmail.com

---

> Dose-Aware Cold Diffusion bridges physics and perception - enabling consistent, dose-robust reconstruction for safer medical imaging.

---

This project is released under the MIT License.
© 2025 DACD Contributors. All rights reserved.
