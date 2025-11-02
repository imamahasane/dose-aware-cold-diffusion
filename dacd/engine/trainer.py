from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader
from rich.progress import Progress
from dacd.models.dacd_model import DACDModel
from dacd.engine.optimizer import build_optimizer
from dacd.engine.schedulers import build_cosine_with_warmup


@dataclass
class TrainCfg:
    data_root: str
    img_hw: tuple[int, int] = (384, 384)
    angles: int = 512
    det: int = 512
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    total_steps: int = 10000
    stage: str = "a"
    amp: bool = True
    out_dir: str = "outputs/stage_a"
    resume: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer:
    def __init__(self, cfg: TrainCfg, loader: DataLoader):
        self.cfg = cfg
        self.loader = loader
        self.model = DACDModel(cfg.img_hw, cfg.angles, cfg.det, device=cfg.device)
        self.optim = build_optimizer(self.model.parameters(), lr=cfg.lr, wd=cfg.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
        self.ckpt_dir = Path(cfg.out_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        if cfg.resume and Path(cfg.resume).exists():
            self._load(cfg.resume)

        self.scheduler = build_cosine_with_warmup(self.optim, cfg.warmup_steps, cfg.total_steps)
        self.global_step = 0
        
    def _save(self, name: str = "last.pt", extras: dict | None = None):
        ckpt = {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.global_step,
            "cfg": self.cfg.__dict__,
        }
        if extras:
            ckpt.update(extras)
        torch.save(ckpt, self.ckpt_dir / name)

    def _load(self, path: str):
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.model.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            self.optim.load_state_dict(ckpt["optim"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt.get("step", 0)

    def fit(self, epochs: int):
        self.model.train()
        best_loss = float("inf")
        with Progress() as progress:
            task = progress.add_task("[green]Training", total=epochs * len(self.loader))
            for ep in range(epochs):
                for batch in self.loader:
                    self.global_step += 1
                    with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                        out = self.model.forward_train(batch)
                        loss = out["loss"]
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.optim.zero_grad(set_to_none=True)
                    self.scheduler.step()

                    progress.advance(task)
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        self._save("best.pt", extras={"best_loss": best_loss})
                self._save("last.pt")
        return best_loss
