from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from torch_radon import Radon
    TORCH_RADON_AVAILABLE = True
except Exception:
    TORCH_RADON_AVAILABLE = False


class CTOps:
    def __init__(
        self,
        img_hw: Tuple[int, int],
        angles: int = 512,
        det_count: Optional[int] = None,
        device: torch.device | str = "cpu",
        use_torch_radon: Optional[bool] = None,
    ):
        self.H, self.W = img_hw
        self.device = torch.device(device)
        self.angles = angles
        self.det_count = det_count or max(self.H, self.W)
        self.use_torch_radon = TORCH_RADON_AVAILABLE if use_torch_radon is None else use_torch_radon

        if self.use_torch_radon and TORCH_RADON_AVAILABLE:
            thetas = torch.linspace(0, math.pi, steps=self.angles, dtype=torch.float32)
            self.radon = Radon(self.H, thetas, det_count=self.det_count, det_spacing=1.0).to(self.device)
        else:
            self.radon = None

    def project(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if self.radon is not None:
            return self.radon.forward(x[:, 0])
        B = x.shape[0]
        thetas = torch.linspace(0, math.pi, steps=self.angles, device=x.device)
        sino = []
        for theta in thetas:
            grid = _affine_rotate_grid(self.H, self.W, theta, device=x.device)
            rotated = F.grid_sample(x, grid, mode="bilinear", align_corners=True)
            sino.append(rotated.sum(dim=2)[:, 0])
        y = torch.stack(sino, dim=1)
        if y.shape[-1] != self.det_count:
            y = F.interpolate(y, size=(self.angles, self.det_count), mode="bilinear", align_corners=True)
        return y

    def backproject(self, y: torch.Tensor) -> torch.Tensor:
        y = y.to(self.device)
        if self.radon is not None:
            img = self.radon.backward(y)
            return img.unsqueeze(1)
        B = y.shape[0]
        thetas = torch.linspace(0, math.pi, steps=self.angles, device=y.device)
        acc = torch.zeros(B, 1, self.H, self.W, device=y.device)
        for i, theta in enumerate(thetas):
            line = y[:, i]
            if line.shape[-1] != self.W:
                line = F.interpolate(line.unsqueeze(1), size=self.W, mode="linear", align_corners=True).squeeze(1)
            bar = line[:, None, None, :].expand(B, 1, self.H, self.W)
            grid = _affine_rotate_grid(self.H, self.W, -theta, device=y.device)
            acc = acc + F.grid_sample(bar, grid, mode="bilinear", align_corners=True)
        return acc / self.angles

    def fbp(self, y: torch.Tensor, filter_name: str = "ram-lak") -> torch.Tensor:
        y = y.to(self.device)
        if self.radon is not None:
            return self.radon.fbp(y).unsqueeze(1)
        B, A, D = y.shape
        freqs = torch.fft.rfftfreq(D, d=1.0).to(y.device)
        ramp = freqs.abs()
        Y = torch.fft.rfft(y, dim=-1)
        Yf = Y * ramp[None, None, :]
        yf = torch.fft.irfft(Yf, n=D, dim=-1)
        return self.backproject(yf)

def _affine_rotate_grid(H: int, W: int, theta: float, device=None) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    c, s = math.cos(theta), math.sin(theta)
    M = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], device=device, dtype=torch.float32)
    grid = F.affine_grid(M.unsqueeze(0), torch.Size((1, 1, H, W)), align_corners=True)
    return grid
