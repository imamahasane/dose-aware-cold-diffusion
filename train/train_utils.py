import torch
import torch.nn as nn
import math

def get_time_embedding(timesteps, embedding_dim, max_period=10000):
    """
    Create sinusoidal time embeddings.
    Same as in original Transformer paper.
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
    exponent = exponent.to(timesteps.device)
    
    emb = timesteps[:, None].float() * exponent.exp()[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    
    if embedding_dim % 2 == 1:  # Zero pad if odd dimension
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    
    return emb

class ConditionEmbedding(nn.Module):
    def __init__(self, time_embed_dim, dose_embed_dim, output_dim):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.dose_embed_dim = dose_embed_dim
        
        self.projection = nn.Sequential(
            nn.Linear(time_embed_dim + dose_embed_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, time_emb, dose_emb):
        combined = torch.cat([time_emb, dose_emb], dim=-1)
        return self.projection(combined)

def dcsa_step_allocation(dose_embedding, min_steps, max_steps):
    """
    Dose-Calibrated Step Allocation
    Allocate more steps for lower dose levels
    """
    # Use the norm of dose embedding as a proxy for dose severity
    dose_severity = torch.norm(dose_embedding, dim=1, keepdim=True)
    dose_severity = (dose_severity - dose_severity.min()) / (dose_severity.max() - dose_severity.min() + 1e-8)
    
    # More steps for higher severity (lower dose)
    steps = min_steps + (max_steps - min_steps) * dose_severity
    return steps.round().long().squeeze()