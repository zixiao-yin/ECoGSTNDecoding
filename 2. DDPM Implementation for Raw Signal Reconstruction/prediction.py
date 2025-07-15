"""
Functions for generating and processing model predictions
"""
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from hydra import compose, initialize

# Import local modules
from utils import clear_gpu_memory, get_frequency_band_power
from hyperparameter_tuning import tune_hyperparameters_safely
from trainer import train_ecog_dbs_model
from visualization import plot_and_save_psd_comparison, analyze_frequency_bands

def get_all_predictions_fast(model, test_dataset, batch_size=2048):
    """Memory-optimized and faster prediction generation"""
    model.eval()
    
    # Use larger batch size and enable cuda optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    # Create DataLoader with optimized settings
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,           # Use multiple workers for data loading
        pin_memory=True,         # Speed up CPU to GPU transfer
        persistent_workers=True,  # Keep workers alive between batches
        prefetch_factor=2        # Prefetch next batches
    )
    
    # Pre-allocate lists with estimated size
    n_samples = len(test_dataset)
    real_dbs_all = []
    imputed_dbs_all = []
    
    print(f"\nProcessing {n_samples} samples in {len(loader)} batches...")
    
    with torch.no_grad(), torch.cuda.amp.autocast():  # Use mixed precision
        for batch_idx, batch in enumerate(loader):
            # Move data to device
            ecog_batch = batch['cond'].to(model.device, non_blocking=True)  # Non-blocking transfer
            real_dbs_batch = batch['signal'].numpy()  # Keep on CPU
            
            # Generate predictions
            imputed_dbs_batch = model.sample(
                num_samples=ecog_batch.size(0),
                cond=ecog_batch,
                noise_type="alpha_beta"
            ).cpu().numpy()
            
            # Store results
            real_dbs_all.append(real_dbs_batch)
            imputed_dbs_all.append(imputed_dbs_batch)
            
            # Clear batch data
            del ecog_batch
            if batch_idx % 5 == 0:  # Reduce frequency of memory clearing
                clear_gpu_memory()
                print(f"Processed {batch_idx + 1}/{len(loader)} batches")
    
    # Concatenate results efficiently
    print("\nProcessing results...")
    real_dbs_all = np.concatenate(real_dbs_all, axis=0).squeeze()
    imputed_dbs_all = np.concatenate(imputed_dbs_all, axis=0).squeeze()
    
    # Calculate band powers more efficiently
    print("\nCalculating frequency band powers...")
    bands = ['theta', 'alpha', 'beta', 'gamma']
    band_powers_real = {band: [] for band in bands}
    band_powers_imputed = {band: [] for band in bands}
    
    # Process band powers in larger chunks
    chunk_size = 100
    for i in range(0, len(real_dbs_all), chunk_size):
        if i % 1000 == 0:
            print(f"Computing band powers: {i}/{len(real_dbs_all)}")
        
        # Process a chunk of signals at once
        chunk_slice = slice(i, min(i + chunk_size, len(real_dbs_all)))
        real_chunk = real_dbs_all[chunk_slice]
        imputed_chunk = imputed_dbs_all[chunk_slice]
        
        # Process each signal in the chunk
        for j in range(len(real_chunk)):
            powers_real = get_frequency_band_power(real_chunk[j])
            powers_imputed = get_frequency_band_power(imputed_chunk[j])
            
            for band in bands:
                band_powers_real[band].append(powers_real[band])
                band_powers_imputed[band].append(powers_imputed[band])
    
    # Convert to numpy arrays efficiently
    for band in bands:
        band_powers_real[band] = np.array(band_powers_real[band])
        band_powers_imputed[band] = np.array(band_powers_imputed[band])
    
    return {
        'real_dbs': real_dbs_all,
        'imputed_dbs': imputed_dbs_all,
        'band_powers_real': band_powers_real,
        'band_powers_imputed': band_powers_imputed
    }

def process_single_subject(ecog_path, dbs_path, output_dir, config_path):
    """Process a single subject's data with detailed hyperparameter tuning"""
    # Extract subject ID outside try block
    subject_id = os.path.basename(ecog_path).split('_')[0]
    side = os.path.basename(ecog_path).split('_')[1][:4]
    full_subject_id = f"{subject_id}_{side}"
    
    try:
        # Create subject-specific output directory
        subject_tuning_dir = os.path.join(output_dir, f"{full_subject_id}_tuning")
        os.makedirs(subject_tuning_dir, exist_ok=True)
        
        print(f"\nProcessing subject {full_subject_id}")
        start_time = time.time()
        
        # Load data
        print("\nLoading data...")
        ecog_data = np.load(ecog_path)
        dbs_data = np.load(dbs_path)
        
        # Initialize config
        with initialize(version_base=None, config_path=config_path):
            base_config = compose(config_name="53_config_ou_200")
        
        # Add signal_channel and cond_dim to network config
        base_config.network.signal_channel = 1
        base_config.network.cond_dim = 3
        
        # Run hyperparameter tuning
        print("\nRunning hyperparameter tuning...")
        best_params, val_dataset, final_config = tune_hyperparameters_safely(
            ecog_data=ecog_data,
            dbs_data=dbs_data,
            base_config=base_config,
            output_dir=subject_tuning_dir,
            n_trials=30,
            timeout=3600*4,
            tuning_epochs=50,
            final_epochs=300
        )
        
        # Create dataset for final training
        from dataset import ECoGDBSDataset
        from torch.utils.data import random_split
        dataset = ECoGDBSDataset(ecog_data, dbs_data)
        train_size = int(0.8 * len(dataset))
        train_dataset, _ = random_split(dataset, [train_size, len(dataset) - train_size])
        
        # Train final model with best parameters
        print("\nTraining final model with best parameters...")
        diffusion_model, _, _ = train_ecog_dbs_model(
            train_dataset=train_dataset,  # Changed this
            val_dataset=val_dataset,      # Changed this
            config=final_config
        )
        
        # Generate predictions
        print("\nGenerating predictions...")
        results = get_all_predictions_fast(diffusion_model, val_dataset, batch_size=2048)
        
        # Plot and save PSD comparison
        plot_and_save_psd_comparison(results, full_subject_id, output_dir)
        
        # Clear model and datasets
        del diffusion_model, train_dataset
        clear_gpu_memory()
        
        # Analyze and save results
        print("\nAnalyzing frequency bands...")
        correlations = analyze_frequency_bands(results)
        
        # Save final results
        results_path = os.path.join(output_dir, f"{subject_id}_{side}_prediction_results.npz")
        print(f"\nSaving results to {results_path}")
        np.savez(results_path,
                 real_dbs=results['real_dbs'],
                 imputed_dbs=results['imputed_dbs'],
                 band_powers_real=results['band_powers_real'],
                 band_powers_imputed=results['band_powers_imputed'],
                 best_hyperparameters=best_params,
                 correlations=correlations)
        
        duration = time.time() - start_time
        print(f"\nCompleted processing {full_subject_id}")
        print(f"Total duration: {duration:.2f} seconds")
        
        return True, duration
        
    except Exception as e:
        print(f"Error processing {full_subject_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        return False, 0