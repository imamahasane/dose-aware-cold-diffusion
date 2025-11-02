import torch
from torch.utils.data import DataLoader, TensorDataset
from dacd.models.dacd_model import DACDModel

def test_smoke_training_step_cpu():
    # Tiny synthetic sample to ensure the forward_train path works.
    B,H,W,A,D = 1,64,64,30,64
    model = DACDModel((H,W), angles=A, det=D, device="cpu")
    batch = {
        "xgt": torch.randn(B,1,H,W),
        "yd": torch.rand(B,A,D),
        "xfbpd": torch.randn(B,1,H,W),
        "dose": torch.tensor([[0.25]]),
        "id": "sample",
    }
    out = model.forward_train(batch)
    assert "loss" in out and torch.isfinite(out["loss"]) and out["xhat"].shape == (B,1,H,W)
