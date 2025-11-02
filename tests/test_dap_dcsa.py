import torch
from dacd.models.dap import DAP, ranking_loss, contrastive_loss
from dacd.models.dcsa import DCSA

def test_dap_outputs_and_losses():
    dap = DAP(in_ch=1, emb_dim=64)
    x = torch.randn(4,1,64,64)
    s,e = dap(x)
    assert s.shape == (4,) and e.shape == (4,64)
    doses = torch.tensor([0.05, 0.125, 0.25, 0.5])
    lrank = ranking_loss(s, doses)
    lcon = contrastive_loss(e, doses)
    assert lrank >= 0 and lcon >= 0

def test_dcsa_range():
    dcsa = DCSA(emb_dim=64, t_min=4, t_max=16)
    e = torch.randn(5,64)
    steps = dcsa(e)
    assert steps.min() >= 4 and steps.max() <= 16
