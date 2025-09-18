import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Image encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(config['input_channels'], config['base_channels'], 3, padding=1),
            nn.InstanceNorm2d(config['base_channels']),
            nn.ReLU(inplace=True),
            nn.Conv2d(config['base_channels'], config['base_channels'], 3, padding=1),
            nn.InstanceNorm2d(config['base_channels']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(config['base_channels'], config['base_channels']*2, 3, padding=1),
            nn.InstanceNorm2d(config['base_channels']*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config['base_channels']*2, config['base_channels']*2, 3, padding=1),
            nn.InstanceNorm2d(config['base_channels']*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(config['base_channels']*2, config['base_channels']*4, 3, padding=1),
            nn.InstanceNorm2d(config['base_channels']*4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Dose prediction head
        self.dose_head = nn.Sequential(
            nn.Linear(config['base_channels']*4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0-1
        )
        
        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(config['base_channels']*4, config['embedding_dim']),
            nn.LayerNorm(config['embedding_dim'])
        )
        
    def forward(self, x):
        features = self.encoder(x).squeeze(-1).squeeze(-1)
        dose_pred = self.dose_head(features)
        embedding = self.embedding_head(features)
        
        # Normalize embedding
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return dose_pred, embedding
    
    def rank_loss(self, embeddings, doses, margin=0.1):
        """Contrastive loss for dose ranking"""
        batch_size = embeddings.size(0)
        loss = 0
        
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                # Determine which dose is higher
                if doses[i] > doses[j]:
                    anchor, positive = embeddings[i], embeddings[j]
                else:
                    anchor, positive = embeddings[j], embeddings[i]
                
                # Cosine similarity
                sim = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0))
                
                # Margin ranking loss
                loss += F.margin_ranking_loss(
                    sim, torch.zeros_like(sim), 
                    torch.ones_like(sim), margin=margin
                )
        
        return loss / (batch_size * (batch_size - 1) / 2) if batch_size > 1 else 0