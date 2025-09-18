import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os

class MayoCTDataset(Dataset):
    def __init__(self, h5_path, split='train', transform=None, dose_levels=None):
        self.h5_path = h5_path
        self.split = split
        self.transform = transform
        
        with h5py.File(h5_path, 'r') as f:
            self.patient_ids = list(f[split].keys())
            self.samples = []
            
            for patient_id in self.patient_ids:
                patient_group = f[split][patient_id]
                for slice_idx in range(patient_group['x_gt'].shape[0]):
                    for dose in (dose_levels if dose_levels else patient_group.attrs['dose_levels']):
                        self.samples.append((patient_id, slice_idx, dose))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        patient_id, slice_idx, dose = self.samples[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            patient_group = f[self.split][patient_id]
            
            x_gt = patient_group['x_gt'][slice_idx]
            y_d = patient_group['y_d'][str(dose)][slice_idx]
            x_fbp = patient_group['x_fbp'][str(dose)][slice_idx]
            
            # Convert to tensors
            x_gt = torch.from_numpy(x_gt).float().unsqueeze(0)
            y_d = torch.from_numpy(y_d).float().unsqueeze(0)
            x_fbp = torch.from_numpy(x_fbp).float().unsqueeze(0)
            dose = torch.tensor(dose).float()
            
            if self.transform:
                x_gt, y_d, x_fbp = self.transform(x_gt, y_d, x_fbp)
            
            return {
                'x_gt': x_gt,
                'y_d': y_d,
                'x_fbp': x_fbp,
                'dose': dose
            }

def get_data_loaders(h5_path, batch_size, num_workers, transform=None):
    train_dataset = MayoCTDataset(h5_path, 'train', transform)
    val_dataset = MayoCTDataset(h5_path, 'val', transform)
    test_dataset = MayoCTDataset(h5_path, 'test', transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader