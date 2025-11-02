from __future__ import annotations
import torch
import torch.nn as nn

def conv_block(c_in, c_out, k=3, s=1, p=1, gn=True):
    layers = [nn.Conv2d(c_in, c_out, k, s, p), nn.GELU()]
    if gn:
        layers.insert(1, nn.GroupNorm(8, c_out))
    return nn.Sequential(*layers)

class UNet2D(nn.Module):
    """Lightweight UNet with optional prior concatenation at input."""
    def __init__(self, c_in=1, c_out=1, base=64):
        super().__init__()
        self.enc1 = nn.Sequential(conv_block(c_in, base), conv_block(base, base))
        self.down1 = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.enc2 = nn.Sequential(conv_block(base * 2, base * 2), conv_block(base * 2, base * 2))
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, 2, 1)
        self.enc3 = nn.Sequential(conv_block(base * 4, base * 4), conv_block(base * 4, base * 4))

        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)
        self.dec2 = nn.Sequential(conv_block(base * 4, base * 2), conv_block(base * 2, base * 2))
        self.up2 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.dec1 = nn.Sequential(conv_block(base * 2, base), conv_block(base, base))

        self.out = nn.Conv2d(base, c_out, 1)

    def forward(self, x, prior=None):
        if prior is not None:
            if x.shape[1] == 1:
                x = torch.cat([x, prior], dim=1)  # inject prior channels
            else:
                x = x  # prior could be injected at deeper layers if desired
        e1 = self.enc1(x)
        e2in = self.down1(e1)
        e2 = self.enc2(e2in)
        e3in = self.down2(e2)
        e3 = self.enc3(e3in)
        d2 = self.up1(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up2(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)
