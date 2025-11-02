from __future__ import annotations
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

def build_cosine_with_warmup(optim, warmup_steps: int, total_steps: int):
    warm = LinearLR(optim, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cos = CosineAnnealingLR(optim, T_max=max(1, total_steps - warmup_steps))
    return SequentialLR(optim, schedulers=[warm, cos], milestones=[warmup_steps])
