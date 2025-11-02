import torch
from dacd.data.ct_ops import CTOps

def test_ct_ops_roundtrip_cpu():
    H, W = 64, 64
    ct = CTOps((H, W), angles=30, det_count=64, device="cpu", use_torch_radon=False)
    x = torch.zeros(1,1,H,W)
    x[:,:,H//4:3*H//4,W//4:3*W//4] = 1.0  # simple square
    y = ct.project(x)
    xr = ct.backproject(y)
    assert y.shape == (1, 30, 64)
    assert xr.shape == (1,1,H,W)
    # Backprojection of projection should correlate with original
    assert torch.corrcoef(torch.stack([x.flatten(), xr.flatten()]))[0,1] > 0.5
