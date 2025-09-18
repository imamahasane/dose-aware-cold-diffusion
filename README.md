# Physics-Guided Cold Diffusion for CT Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

An implementation of physics-guided cold diffusion with FBP warm start for low-dose CT reconstruction, based on the Mayo Clinic 2020 dataset.

## Overview

This project implements a novel cold diffusion approach for CT reconstruction that uses filtered backprojection (FBP) warm start for initialization, incorporates physics-based data consistency steps, and employs dose-aware conditioning and step allocation (DCSA).

## Key Features

- **Dose-aware perception module** for continuous dose embedding
- **Physics-consistent diffusion** with data consistency steps
- **FBP warm start** for faster convergence and improved stability
- **DCSA scheduling** for dose-adaptive step allocation
- **Uncertainty-guided regularization** for improved stability
- **Two-stage training** (perception + reconstruction)
- **Comprehensive evaluation** with multiple metrics (PSNR, SSIM, NPS)

## Installation

```bash
git clone https://github.com/your-username/ColdDiffusionCT.git
cd ColdDiffusionCT
pip install -r requirements.txt
```
## Dataset Preparation
1. Download the Mayo Clinic 2020 Low-Dose CT Grand Challenge dataset
2. Preprocess data using the provided scripts:
```bash
python scripts/preprocess_data.py --input_dir /path/to/mayo/data --output_dir /path/to/processed/data
```
## Training
Stage A: Train Perception Module
```bash
python scripts/train_stage_a.py --config configs/default.yaml --data_path /path/to/training_data.h5
```
Stage B: Train Reconstruction Diffusion
```bash
python scripts/train_stage_b.py --config configs/default.yaml --data_path /path/to/training_data.h5
```
## Evaluation
```bash
python scripts/evaluate.py --checkpoint path/to/checkpoint.pth --data_path /path/to/test_data.h5
```

## Project Structure
```bash
physics-guided-cold-diffusion/
├── configs/           # Configuration files
├── data/              # Data loading utilities
├── models/            # Model architectures
├── ops/               # Physics operations and projections
├── train/             # Training routines
├── eval/              # Evaluation scripts
├── scripts/           # Utility scripts
└── tests/             # Unit tests
```

## Results
Coming soon....

## Contact
Md Imam Ahasan - emamahasane@gmail.com
