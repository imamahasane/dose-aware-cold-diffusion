from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

@dataclass
class DataCfg:
    root: str
    patch_hw: Tuple[int, int] = (384, 384)
    hu_clip: Tuple[int, int] = (-1024, 3072)
    doses: Tuple[float, ...] = (0.5, 0.25, 0.125, 0.05)
    stage: str = "a"

class SimpleCTDataset(Dataset): 
    def __init__(self, cfg: DataCfg, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.root = Path(cfg.root)
        self.images = sorted((self.root / "images").glob("*.pt"))
        assert len(self.images) > 0, f"No images found in {self.root}/images"
        self.dose_dirs = {d: (self.root / f"sinograms_d{d}") for d in cfg.doses}
        self.fbp_dirs = {d: (self.root / f"fbp_d{d}") for d in cfg.doses}

    def __len__(self) -> int:
        return len(self.images) * len(self.cfg.doses)

    def __getitem__(self, idx: int):
        n = len(self.images)
        img_idx = idx % n
        dose_idx = idx // n
        dose = self.cfg.doses[dose_idx]

        xgt = torch.load(self.images[img_idx])  # [1,H,W], normalized [-1,1]
        sino = torch.load(self.dose_dirs[dose] / self.images[img_idx].name)  # [A,D]
        xfbp = torch.load(self.fbp_dirs[dose] / self.images[img_idx].name)  # [1,H,W]

        return {
            "xgt": xgt.float(),
            "yd": sino.float(),
            "xfbpd": xfbp.float(),
            "dose": torch.tensor([dose], dtype=torch.float32),
            "id": self.images[img_idx].name,
        }

def make_loader(cfg: DataCfg, split: str, batch_size: int, num_workers: int = 8, shuffle=True):
    ds = SimpleCTDataset(cfg, split=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
