import torch
from dacd.eval.metrics import psnr, ssim, rmse

def test_metrics_shapes():
    x = torch.zeros(2,1,32,32)
    y = torch.zeros(2,1,32,32)
    assert psnr(x,y).shape == (2,)
    assert ssim(x,y).shape == (2,)
    assert rmse(x,y).shape == (2,)
