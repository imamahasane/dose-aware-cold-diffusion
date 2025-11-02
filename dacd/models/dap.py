from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DAP(nn.Module):
    """Dose-Aware Perception: predicts severity score and dose embedding from x_t."""
    def __init__(self, in_ch=1, emb_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_vec = nn.Flatten()
        self.fc_sev = nn.Linear(128, 1)
        self.fc_emb = nn.Linear(128, emb_dim)

    def forward(self, x):
        h = self.enc(x)
        v = self.to_vec(h)
        s = self.fc_sev(v)
        e = F.normalize(self.fc_emb(v), dim=-1)
        return s.squeeze(-1), e

def ranking_loss(scores: torch.Tensor, doses: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    N = scores.shape[0]
    loss = 0.0
    count = 0
    for i in range(N):
        for j in range(N):
            if doses[i] < doses[j]:
                loss = loss + torch.clamp(margin - (scores[j] - scores[i]), min=0.0)
                count += 1
    return loss / max(count, 1)

def contrastive_loss(z: torch.Tensor, labels: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    with torch.no_grad():
        q = torch.quantile(labels, torch.tensor([0.25, 0.5, 0.75], device=labels.device))
        bins = torch.bucketize(labels, q)
    sim = z @ z.t()
    sim = sim / tau
    eye = torch.eye(sim.size(0), device=sim.device)
    sim = sim - 1e9 * eye
    pos = (bins[:, None] == bins[None, :]).float() - eye
    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    loss = -(log_prob * pos).sum() / torch.clamp(pos.sum(), min=1.0)
    return loss
