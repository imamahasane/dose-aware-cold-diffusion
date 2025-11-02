from __future__ import annotations
import torch

def build_optimizer(params, lr=1e-4, wd=1e-2, betas=(0.9, 0.999)):
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)
