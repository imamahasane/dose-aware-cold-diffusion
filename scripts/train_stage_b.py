import torch
import yaml
from models import PerceptionModule, ReconstructionUNet, PEMPlus
from ops import PhysicsOperator
from data import get_data_loaders
from train import ColdDiffusionTrainer

def main():
    # Load config
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    perception_model = PerceptionModule(config['model']['perception'])
    reconstruction_model = ReconstructionUNet(config['model']['reconstruction'])
    pem_model = PEMPlus(config['model'].get('pem', {}))
    
    # Initialize physics operator
    physics_operator = PhysicsOperator(config['physics']['geometry'])
    
    # Load data
    train_loader, val_loader, _ = get_data_loaders(
        'path/to/data.h5',
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # Initialize trainer
    trainer = ColdDiffusionTrainer(
        perception_model, reconstruction_model, pem_model,
        physics_operator, config, device
    )
    
    # Load pre-trained perception model
    perception_checkpoint = torch.load('checkpoints/perception_best.pth')
    trainer.perception.load_state_dict(perception_checkpoint['model_state_dict'])
    
    # Train reconstruction model
    trainer.train_stage_b(train_loader, val_loader, config['training']['num_epochs'])
    
    # Save model
    torch.save({
        'model_state_dict': trainer.reconstruction.state_dict(),
        'config': config
    }, 'checkpoints/reconstruction_final.pth')

if __name__ == '__main__':
    main()