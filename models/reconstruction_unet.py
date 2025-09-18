import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.InstanceNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        out += residual
        return F.relu(out)

class ReconstructionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initial projection for conditioning
        self.condition_proj = nn.Linear(config['conditioning_dim'], config['base_channels'] * 8)
        
        # Encoder
        self.enc1 = ResidualBlock(config['input_channels'], config['base_channels'])
        self.enc2 = ResidualBlock(config['base_channels'], config['base_channels'] * 2, stride=2)
        self.enc3 = ResidualBlock(config['base_channels'] * 2, config['base_channels'] * 4, stride=2)
        self.enc4 = ResidualBlock(config['base_channels'] * 4, config['base_channels'] * 8, stride=2)
        
        # Decoder
        self.dec3 = ResidualBlock(config['base_channels'] * 12, config['base_channels'] * 4)  # Skip connection
        self.dec2 = ResidualBlock(config['base_channels'] * 6, config['base_channels'] * 2)   # Skip connection
        self.dec1 = ResidualBlock(config['base_channels'] * 3, config['base_channels'])        # Skip connection
        
        # Final convolution
        self.final = nn.Conv2d(config['base_channels'], 1, 1)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x, pem_features, conditioning):
        # Process conditioning
        cond_features = self.condition_proj(conditioning)
        gamma, beta = torch.chunk(cond_features, 2, dim=1)
        gamma = gamma.view(-1, self.config['base_channels'] * 8, 1, 1)
        beta = beta.view(-1, self.config['base_channels'] * 8, 1, 1)
        
        # Encoder
        e1 = self.enc1(x)  # base_channels
        e2 = self.enc2(e1)  # base_channels * 2
        e3 = self.enc3(e2)  # base_channels * 4
        e4 = self.enc4(e3)  # base_channels * 8
        
        # Apply conditioning
        e4 = e4 * (1 + gamma) + beta
        
        # Decoder with skip connections
        d3 = self.upsample(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # base_channels * 4
        
        d2 = self.upsample(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # base_channels * 2
        
        d1 = self.upsample(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # base_channels
        
        # Add PEM features
        if pem_features is not None:
            # Ensure pem_features has the same spatial size
            if pem_features.size(2) != d1.size(2) or pem_features.size(3) != d1.size(3):
                pem_features = F.interpolate(pem_features, size=d1.shape[2:], mode='bilinear', align_corners=True)
            d1 = d1 + pem_features
        
        return self.final(d1)