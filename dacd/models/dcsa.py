from __future__ import annotations
import torch
import torch.nn as nn

class DCSA(nn.Module):
    """Dose-Calibrated Step Allocation."""
    def __init__(self, emb_dim=64, t_min=4, t_max=16):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 1)
        self.t_min = t_min
        self.t_max = t_max

    def forward(self, e_d: torch.Tensor) -> torch.Tensor:
        sev = torch.sigmoid(self.fc(e_d)).squeeze(-1)
        steps = self.t_max - torch.floor(sev * (self.t_max - self.t_min) + 1e-6)
        steps = steps.clamp(min=self.t_min, max=self.t_max).to(torch.int64)
        return steps
