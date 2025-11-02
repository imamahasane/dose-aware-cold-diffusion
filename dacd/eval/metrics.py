from __future__ import annotations
import torch
import torch.nn.functional as F


def psnr(xhat: torch.Tensor, xgt: torch.Tensor, L: float = 2.0) -> torch.Tensor:
    mse = F.mse_loss(xhat, xgt, reduction="none").flatten(1).mean(dim=1)
    return 10.0 * torch.log10((L ** 2) / (mse + 1e-12))

def ssim(xhat: torch.Tensor, xgt: torch.Tensor, C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
    from torch.nn.functional import conv2d
    device = xhat.device
    gauss = torch.tensor([1, 4, 7, 10, 13, 10, 7, 4, 1], dtype=torch.float32, device=device)
    gauss = gauss[:, None] @ gauss[None, :]
    gauss = gauss / gauss.sum()
    kernel = gauss[None, None, :, :]
    mu_x = conv2d(xhat, kernel, padding=4)
    mu_y = conv2d(xgt, kernel, padding=4)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x2 = conv2d(xhat * xhat, kernel, padding=4) - mu_x2
    sigma_y2 = conv2d(xgt * xgt, kernel, padding=4) - mu_y2
    sigma_xy = conv2d(xhat * xgt, kernel, padding=4) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.flatten(1).mean(dim=1)

def rmse(xhat: torch.Tensor, xgt: torch.Tensor) -> torch.Tensor:
    e2 = (xhat - xgt).pow(2).flatten(1).mean(dim=1)
    return torch.sqrt(e2 + 1e-12)
