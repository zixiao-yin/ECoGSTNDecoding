"""
Utility functions for memory management, timing, and general helpers
"""
import torch
import gc
import numpy as np
from datetime import datetime, timedelta

def clear_gpu_memory():
    """Enhanced GPU memory clearing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

def get_gpu_memory_info():
    """Get current GPU memory usage in GB"""
    if not torch.cuda.is_available():
        return 0, 0
    return (
        torch.cuda.memory_allocated() / 1e9,  # dedicated
        torch.cuda.memory_reserved() / 1e9     # total (dedicated + shared)
    )

def estimate_model_memory(hidden_channel, num_blocks, batch_size, signal_length):
    """Estimate model memory requirements in GB"""
    # Model parameters
    params_per_block = (
        (hidden_channel * hidden_channel * 3) +  # Main conv layers
        (hidden_channel * hidden_channel * num_blocks * 3) +  # SLConv parameters
        (hidden_channel * 4)  # Batch norms
    )
    total_params = params_per_block * num_blocks
    
    # Forward pass activations
    activations_per_block = batch_size * hidden_channel * signal_length * 2
    total_activations = activations_per_block * num_blocks
    
    # Other memory requirements
    optimizer_states = total_params * 2
    gradients = total_params
    backward_activations = total_activations * 1.5
    pytorch_overhead = 0.5 * (1024**3)
    
    total_memory = (
        total_params + 
        total_activations + 
        optimizer_states + 
        gradients + 
        backward_activations
    ) * 4 + pytorch_overhead
    
    return (total_memory / (1024**3)) * 1.2  # 20% buffer

def format_time(seconds):
    """Convert seconds to human-readable format"""
    return str(timedelta(seconds=int(seconds)))

def get_frequency_band_power(signal_data, fs=250, nperseg=256):
    """Calculate power in different frequency bands"""
    from scipy import signal
    
    f, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg)
    
    bands = {
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 35),
        'gamma': (35, 100)
    }
    
    powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        mask = (f >= low_freq) & (f <= high_freq)
        powers[band_name] = np.trapz(psd[mask], f[mask])
    
    return powers

def get_safe_parameter_bounds(available_memory_gb):
    """Get parameter bounds targeting 70-85% GPU utilization"""
    return {
        'hidden_channel': (48, 92),
        'num_blocks': (3, 4),
        'batch_size': (384, 768),
        'diffusion_steps': (500, 1000)
    }