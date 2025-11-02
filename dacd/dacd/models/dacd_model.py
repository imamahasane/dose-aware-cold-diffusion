from __future__ import annotations
import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dacd.data.ct_ops import CTOps
from dacd.models.unet_backbone import UNet2D
from dacd.models.dap import DAP, ranking_loss, contrastive_loss
from dacd.models.pempp import PEMPP
from dacd.models.dcsa import DCSA


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class DACDModel(nn.Module):
    def __init__(self, img_hw: Tuple[int, int], angles: int, det: int, device: str = "cuda"):
        super().__init__()
        self.device_name = device
        self.ct = CTOps(img_hw, angles=angles, det_count=det, device=device)
        self.unet = UNet2D(c_in=1, c_out=1, base=64)
        self.dap = DAP(in_ch=1, emb_dim=64)
        self.pempp = PEMPP(in_ch=1, prior_ch=32, emb_dim=64)
        self.dcsa = DCSA(emb_dim=64, t_min=4, t_max=16)
        self.time_dim = 64
        self.to(self.device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        xgt = batch["xgt"].to(self.device)
        yd = batch["yd"].to(self.device)
        xfbp = batch["xfbpd"].to(self.device)
        dose = batch["dose"].to(self.device).squeeze(-1)

        sev, ed = self.dap(xfbp)
        steps = self.dcsa(ed)
        T = int(steps.max().item())
        xt = xfbp
        for tt in range(T, 0, -1):
            tvec = torch.full((xfbpt := xfbp.shape[0],), float(tt), device=self.device)
            et = timestep_embedding(tvec, self.time_dim)
            prior = self.pempp(xt, ed, et)
            denoised = self.unet(xt, prior=prior)
            proj = self.ct.project(denoised)
            resid = yd - proj
            corr = self.ct.backproject(resid)
            eta = 0.1
            xt = denoised + eta * corr

        xhat = xt
        y_clean = self.ct.project(xgt)
        l_img = F.l1_loss(xhat, xgt) + 0.1 * F.mse_loss(self.ct.project(xhat), y_clean)

        l_rank = ranking_loss(sev, dose, margin=0.1)
        l_con = contrastive_loss(ed, dose, tau=0.07)
        l_total = l_img + 0.1 * l_rank + 0.1 * l_con
        return {"loss": l_total, "l_img": l_img, "l_rank": l_rank, "l_con": l_con, "xhat": xhat}

    @torch.no_grad()
    def infer(self, xfbp: torch.Tensor, yd: torch.Tensor) -> torch.Tensor:
        xfbp = xfbp.to(self.device)
        yd = yd.to(self.device)
        sev, ed = self.dap(xfbp)
        steps = self.dcsa(ed)
        T = int(steps.max().item())
        xt = xfbp
        for tt in range(T, 0, -1):
            tvec = torch.full((xfbpt := xfbp.shape[0],), float(tt), device=self.device)
            et = timestep_embedding(tvec, self.time_dim)
            prior = self.pempp(xt, ed, et)
            denoised = self.unet(xt, prior=prior)
            proj = self.ct.project(denoised)
            resid = yd - proj
            corr = self.ct.backproject(resid)
            eta = 0.1
            xt = denoised + eta * corr
        return xt
