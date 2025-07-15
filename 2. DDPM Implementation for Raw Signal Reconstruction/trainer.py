"""
Training related classes and functions
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from utils import clear_gpu_memory
from dataset import ECoGDBSDataset
from ntd.diffusion_model import Diffusion
from ntd.networks import AdaConv
from ntd.utils.kernels_and_diffusion_utils import OUProcess

class CustomTrainer:
    def __init__(self, model, data_loader, optimizer, scheduler, device):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()
    
    @torch.cuda.amp.autocast()
    def train_step(self, sig_batch, cond_batch):
        loss = self.model.train_batch(sig_batch, cond=cond_batch)
        return torch.mean(loss)
    
    def train_epoch(self):
        self.model.train()
        losses = []
        
        # Remove tqdm wrapper, use simple iteration
        for batch_idx, batch in enumerate(self.data_loader):
            sig_batch = batch["signal"].to(self.device, non_blocking=True)
            cond_batch = batch["cond"].to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                loss = self.train_step(sig_batch, cond_batch)
            
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()
            
            batch_size = sig_batch.shape[0]
            losses.append((batch_size, loss.item()))
            
            if batch_idx % 50 == 49:
                clear_gpu_memory()
                
                # Optional: print summary every 50 batches
                avg_loss = sum(l * s for s, l in losses[-50:]) / sum(s for s, _ in losses[-50:])
                print(f"Batch {batch_idx + 1}/{len(self.data_loader)}, "
                      f"Loss: {avg_loss:.6f}, "
                      f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        return losses

def train_ecog_dbs_model(train_dataset, val_dataset, config):
    """Main training function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.optimizer.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize model components
    network = AdaConv(
        signal_length=config.dataset.signal_length,
        signal_channel=config.network.signal_channel,
        cond_dim=config.network.cond_dim,
        hidden_channel=config.network.hidden_channel,
        in_kernel_size=config.network.in_kernel_size,
        out_kernel_size=config.network.out_kernel_size,
        slconv_kernel_size=config.network.slconv_kernel_size,
        num_scales=config.network.num_scales,
        num_blocks=config.network.num_blocks,
        num_off_diag=config.network.num_off_diag,
        use_pos_emb=config.network.use_pos_emb,
        padding_mode=config.network.padding_mode,
        use_fft_conv=config.network.use_fft_conv,
    ).to(device)
    
    ou_process = OUProcess(
        config.diffusion_kernel.sigma_squared,
        config.diffusion_kernel.ell,
        config.dataset.signal_length
    ).to(device)
    
    diffusion = Diffusion(
        network=network,
        noise_sampler=ou_process,
        mal_dist_computer=ou_process,
        diffusion_time_steps=config.diffusion.diffusion_steps,
        schedule=config.diffusion.schedule,
        start_beta=config.diffusion.start_beta,
        end_beta=config.diffusion.end_beta,
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.optimizer.lr,
        epochs=config.optimizer.num_epochs,
        steps_per_epoch=len(train_loader)
    )
    
    trainer = CustomTrainer(diffusion, train_loader, optimizer, scheduler, device)
    
    # Training loop
    for epoch in range(config.optimizer.num_epochs):
        losses = trainer.train_epoch()
        
        # Calculate epoch statistics
        total_loss = sum(loss * size for size, loss in losses)
        total_samples = sum(size for size, _ in losses)
        epoch_loss = total_loss / total_samples
        
        print(f"\nEpoch {epoch} summary:")
        print(f"Average loss: {epoch_loss:.6f}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
    
    return diffusion, train_dataset, val_dataset