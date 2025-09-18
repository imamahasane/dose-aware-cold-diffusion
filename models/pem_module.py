import torch
import torch.nn as nn
import torch.nn.functional as F

class PEMPlus(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Edge detection kernels
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Laplacian kernel
        self.register_buffer('laplacian', torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Shallow CNN prior
        self.cnn_prior = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1)
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Conv2d(64 + 3, 128, 1),  # CNN features + edge features
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Edge features
        edges_x = F.conv2d(x, self.sobel_x, padding=1)
        edges_y = F.conv2d(x, self.sobel_y, padding=1)
        edges_magnitude = torch.sqrt(edges_x**2 + edges_y**2 + 1e-6)
        
        # Laplacian features
        laplacian = F.conv2d(x, self.laplacian, padding=1)
        
        # CNN features
        cnn_features = self.cnn_prior(x)
        
        # Combine features
        edge_features = torch.cat([edges_magnitude, edges_x, edges_y], dim=1)
        combined = torch.cat([cnn_features, edge_features], dim=1)
        
        # Fuse features
        fused = self.fusion(combined)
        
        return fused
    
    def apply_film(self, features, conditioning):
        """Apply FiLM conditioning to features"""
        gamma, beta = torch.chunk(conditioning, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        return features * (1 + gamma) + beta