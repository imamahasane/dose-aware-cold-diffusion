import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class ColdDiffusionTrainer:
    def __init__(self, perception_model, reconstruction_model, pem_model, 
                 physics_operator, config, device):
        self.perception = perception_model.to(device)
        self.reconstruction = reconstruction_model.to(device)
        self.pem = pem_model.to(device)
        self.physics = physics_operator
        self.config = config
        self.device = device
        
        # Optimizers
        self.perception_optimizer = optim.AdamW(
            self.perception.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.reconstruction_optimizer = optim.AdamW(
            self.reconstruction.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Loss functions
        self.reconstruction_loss = nn.L1Loss()
        self.rank_loss = nn.MarginRankingLoss(margin=0.1)
        
        # Condition embedding
        self.condition_embedding = ConditionEmbedding(
            time_embed_dim=64,
            dose_embed_dim=config['model']['perception']['embedding_dim'],
            output_dim=config['model']['reconstruction']['conditioning_dim']
        ).to(device)
        
    def train_stage_a(self, train_loader, val_loader, num_epochs):
        """Train perception module"""
        self.perception.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                x_fbp, dose = batch['x_fbp'], batch['dose']
                x_fbp, dose = x_fbp.to(self.device), dose.to(self.device)
                
                self.perception_optimizer.zero_grad()
                
                # Forward pass
                dose_pred, dose_emb = self.perception(x_fbp)
                
                # Losses
                mse_loss = nn.MSELoss()(dose_pred.squeeze(), dose)
                rank_loss = self.perception.rank_loss(dose_emb, dose)
                loss = mse_loss + 0.1 * rank_loss
                
                # Backward pass
                loss.backward()
                self.perception_optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validation
            val_loss = self.validate_stage_a(val_loader)
            print(f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}, Val Loss = {val_loss:.4f}")
    
    def train_stage_b(self, train_loader, val_loader, num_epochs):
        """Train reconstruction diffusion"""
        # Freeze perception module
        self.perception.eval()
        
        for epoch in range(num_epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                x_gt, y_d, x_fbp, dose = batch['x_gt'], batch['y_d'], batch['x_fbp'], batch['dose']
                x_gt, y_d, x_fbp, dose = [t.to(self.device) for t in [x_gt, y_d, x_fbp, dose]]
                
                self.reconstruction_optimizer.zero_grad()
                
                # Get dose embedding (no gradients for perception)
                with torch.no_grad():
                    _, dose_emb = self.perception(x_fbp)
                
                # DCSA: Determine number of steps
                T_x = dcsa_step_allocation(
                    dose_emb, 
                    self.config['training']['min_steps'],
                    self.config['training']['max_steps']
                )
                
                # Initialize with FBP warm start
                z_t = x_fbp
                
                # Diffusion process
                for step in range(T_x):
                    # Time embedding
                    time_emb = get_time_embedding(
                        torch.tensor([step/T_x], device=self.device).repeat(x_gt.size(0)),
                        embedding_dim=64
                    )
                    
                    # Condition embedding
                    conditioning = self.condition_embedding(time_emb, dose_emb)
                    
                    # PEM features
                    pem_features = self.pem(z_t)
                    
                    # Denoising step
                    z_pred = self.reconstruction(z_t, pem_features, conditioning)
                    
                    # Physics consistency step
                    eta = self.config['training']['eta_schedule'][min(step, len(self.config['training']['eta_schedule'])-1)]
                    z_t = self.physics_step(z_pred, y_d, eta)
                
                # Final loss
                loss = self.reconstruction_loss(z_t, x_gt)
                
                # Backward pass
                loss.backward()
                self.reconstruction_optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validation
            val_psnr = self.validate_stage_b(val_loader)
            print(f"Epoch {epoch+1}: Train Loss = {total_loss/len(train_loader):.4f}, Val PSNR = {val_psnr:.2f}")
    
    def physics_step(self, z_pred, y_d, eta):
        """Physics data consistency step"""
        residual = self.physics.forward_op(z_pred) - y_d
        return z_pred - eta * self.physics.backward_op(residual)
    
    def validate_stage_a(self, val_loader):
        self.perception.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_fbp, dose = batch['x_fbp'], batch['dose']
                x_fbp, dose = x_fbp.to(self.device), dose.to(self.device)
                
                dose_pred, dose_emb = self.perception(x_fbp)
                mse_loss = nn.MSELoss()(dose_pred.squeeze(), dose)
                rank_loss = self.perception.rank_loss(dose_emb, dose)
                loss = mse_loss + 0.1 * rank_loss
                
                total_loss += loss.item()
        
        self.perception.train()
        return total_loss / len(val_loader)
    
    def validate_stage_b(self, val_loader):
        self.reconstruction.eval()
        total_psnr = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x_gt, y_d, x_fbp, dose = batch['x_gt'], batch['y_d'], batch['x_fbp'], batch['dose']
                x_gt, y_d, x_fbp, dose = [t.to(self.device) for t in [x_gt, y_d, x_fbp, dose]]
                
                # Get dose embedding
                _, dose_emb = self.perception(x_fbp)
                
                # DCSA
                T_x = dcsa_step_allocation(
                    dose_emb, 
                    self.config['training']['min_steps'],
                    self.config['training']['max_steps']
                )
                
                # Reconstruction
                z_t = x_fbp
                for step in range(T_x):
                    time_emb = get_time_embedding(
                        torch.tensor([step/T_x], device=self.device).repeat(x_gt.size(0)),
                        embedding_dim=64
                    )
                    conditioning = self.condition_embedding(time_emb, dose_emb)
                    pem_features = self.pem(z_t)
                    z_pred = self.reconstruction(z_t, pem_features, conditioning)
                    
                    eta = self.config['training']['eta_schedule'][min(step, len(self.config['training']['eta_schedule'])-1)]
                    z_t = self.physics_step(z_pred, y_d, eta)
                
                # Calculate PSNR
                mse = nn.MSELoss()(z_t, x_gt)
                psnr = 10 * torch.log10(1 / mse)
                total_psnr += psnr.item()
        
        self.reconstruction.train()
        return total_psnr / len(val_loader)