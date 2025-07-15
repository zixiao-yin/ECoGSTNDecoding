"""
Dataset classes for handling ECoG and DBS data
"""
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

class ECoGDBSDataset(Dataset):
    def __init__(self, ecog_data, dbs_data):
        """
        Args:
            ecog_data: numpy array of shape (num_epochs, 3, sequence_length)
            dbs_data: numpy array of shape (num_epochs, 1, sequence_length)
        """
        if isinstance(ecog_data, torch.Tensor):
            ecog_data = ecog_data.numpy()
        if isinstance(dbs_data, torch.Tensor):
            dbs_data = dbs_data.numpy()
        
        if len(ecog_data.shape) != 3 or len(dbs_data.shape) != 3:
            raise ValueError("Data should be 3D: (epochs, channels, sequence_length)")
        if ecog_data.shape[0] != dbs_data.shape[0]:
            raise ValueError("Number of epochs don't match")
        if ecog_data.shape[2] != dbs_data.shape[2]:
            raise ValueError("Sequence lengths don't match")
        
        self.ecog_data = np.ascontiguousarray(ecog_data, dtype=np.float32)
        self.dbs_data = np.ascontiguousarray(dbs_data, dtype=np.float32)
        
        # Compute normalization statistics
        self.ecog_mean = np.mean(self.ecog_data, axis=(0, 2), keepdims=True)
        self.ecog_std = np.std(self.ecog_data, axis=(0, 2), keepdims=True)
        self.dbs_mean = np.mean(self.dbs_data, axis=(0, 2), keepdims=True)
        self.dbs_std = np.std(self.dbs_data, axis=(0, 2), keepdims=True)
        
        # Prevent division by zero
        self.ecog_std[self.ecog_std == 0] = 1
        self.dbs_std[self.dbs_std == 0] = 1
    
    def __len__(self):
        return len(self.ecog_data)
    
    def __getitem__(self, idx):
        ecog_sample = (self.ecog_data[idx] - self.ecog_mean[0]) / self.ecog_std[0]
        dbs_sample = (self.dbs_data[idx] - self.dbs_mean[0]) / self.dbs_std[0]
        
        return {
            "signal": torch.from_numpy(dbs_sample).float(),
            "cond": torch.from_numpy(ecog_sample).float()
        }

def temporal_train_test_split(dataset, train_ratio=0.8):
    """Split dataset while preserving temporal order."""
    total_length = len(dataset)
    train_size = int(total_length * train_ratio)
    
    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, total_length))
    
    print(f"\nTemporal Split Information:")
    print(f"Total samples: {total_length}")
    print(f"Training samples: {len(train_indices)} (indices 0 to {train_size-1})")
    print(f"Testing samples: {len(test_indices)} (indices {train_size} to {total_length-1})")
    
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, test_dataset