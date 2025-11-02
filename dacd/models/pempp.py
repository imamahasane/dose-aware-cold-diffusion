from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K

class PEMPP(nn.Module):
    """Prior Extraction Module ++: gradient, laplacian, NLM-proxy, CNN prior; attention + FiLM."""
    def __init__(self, in_ch=1, prior_ch=32, emb_dim=64):
        super().__init__()
        self.cnn_prior = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GELU(),
        )
        self.fq = nn.Conv2d(in_ch, prior_ch, 1)
        self.fk = nn.ModuleList([nn.Conv2d(c, prior_ch, 1) for c in [1, 1, 1, 32]])
        self.fv = nn.ModuleList([nn.Conv2d(c, prior_ch, 1) for c in [1, 1, 1, 32]])
        self.film = nn.Sequential(nn.Linear(emb_dim * 2, prior_ch * 2))

    def _priors(self, x: torch.Tensor):
        grad = K.filters.sobel(x)
        grad_mag = torch.linalg.vector_norm(grad, dim=1, keepdim=True)
        lap = K.filters.laplacian(x, kernel_size=3)
        nlm = K.filters.bilateral_blur(x, (5, 5), sigma_color=0.1, sigma_space=1.5)
        cnn = self.cnn_prior(x)
        return grad_mag, lap, nlm, cnn

    def forward(self, x: torch.Tensor, dose_emb: torch.Tensor, time_emb: torch.Tensor):
        grad, lap, nlm, cnn = self._priors(x)
        Q = self.fq(x)
        keys = [self.fk[0](grad), self.fk[1](lap), self.fk[2](nlm), self.fk[3](cnn)]
        vals = [self.fv[0](grad), self.fv[1](lap), self.fv[2](nlm), self.fv[3](cnn)]

        att = []
        for Kk in keys:
            att.append((Q * Kk).sum(dim=1, keepdim=True))
        A = torch.softmax(torch.cat(att, dim=1), dim=1)
        Pfuse = 0.0
        for i, Vk in enumerate(vals):
            Pfuse = Pfuse + A[:, i : i + 1] * Vk

        ed_et = torch.cat([dose_emb, time_emb], dim=-1)
        gb = self.film(ed_et)
        C = Pfuse.shape[1]
        gamma, beta = gb[:, :C], gb[:, C:]
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        Pmod = gamma * Pfuse + beta
        return Pmod
