"""
Visualization and analysis functions
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
from utils import get_frequency_band_power

def plot_and_save_psd_comparison(results, subject_id, output_dir):
    """Plot and save PSD comparison for a subject without progress bars"""
    try:
        # Create figure directory
        fig_dir = os.path.join(output_dir, 'psd_figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        print("\nCalculating average PSDs...")
        real_dbs = results['real_dbs']
        imputed_dbs = results['imputed_dbs']
        
        # Calculate PSDs for all trials
        f, psd_real_all = signal.welch(real_dbs[0], fs=250, nperseg=256)
        psd_imputed_all = signal.welch(imputed_dbs[0], fs=250, nperseg=256)[1]
        
        total_samples = len(real_dbs)
        for i in range(1, total_samples):
            if i % 2000 == 0:  # Print progress less frequently
                print(f"Processing PSDs: {i}/{total_samples}")
                
            psd_real = signal.welch(real_dbs[i], fs=250, nperseg=256)[1]
            psd_imputed = signal.welch(imputed_dbs[i], fs=250, nperseg=256)[1]
            psd_real_all = np.vstack((psd_real_all, psd_real))
            psd_imputed_all = np.vstack((psd_imputed_all, psd_imputed))
        
        # Calculate mean and std
        psd_real_mean = np.mean(psd_real_all, axis=0)
        psd_real_std = np.std(psd_real_all, axis=0)
        psd_imputed_mean = np.mean(psd_imputed_all, axis=0)
        psd_imputed_std = np.std(psd_imputed_all, axis=0)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot real DBS
        plt.semilogy(f, psd_real_mean, 'b', label='Real DBS (mean)', alpha=0.7)
        plt.fill_between(f, 
                        psd_real_mean - psd_real_std, 
                        psd_real_mean + psd_real_std, 
                        color='b', alpha=0.2)
        
        # Plot imputed DBS
        plt.semilogy(f, psd_imputed_mean, 'r', label='Imputed DBS (mean)', alpha=0.7)
        plt.fill_between(f, 
                        psd_imputed_mean - psd_imputed_std, 
                        psd_imputed_mean + psd_imputed_std, 
                        color='r', alpha=0.2)
        
        # Add frequency band annotations
        bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # Add frequency band shading and labels
        colors = ['lightgray', 'lightblue', 'lightgreen', 'lightpink']
        for (band, (fmin, fmax)), color in zip(bands.items(), colors):
            plt.axvspan(fmin, fmax, color=color, alpha=0.2)
            plt.text((fmin + fmax)/2, plt.ylim()[0]*1.1, band, 
                    horizontalalignment='center', verticalalignment='bottom')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title(f'Average Power Spectral Density with Std Dev\n{subject_id}')
        plt.xlim(0, 125)
        plt.legend()
        plt.grid(True)
        
        # Save figure
        fig_path = os.path.join(fig_dir, f'{subject_id}_psd_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save PSD data
        psd_data_path = os.path.join(fig_dir, f'{subject_id}_psd_data.npz')
        np.savez(psd_data_path,
                 frequencies=f,
                 psd_real_mean=psd_real_mean,
                 psd_real_std=psd_real_std,
                 psd_imputed_mean=psd_imputed_mean,
                 psd_imputed_std=psd_imputed_std)
        
        print(f"\nPSD comparison saved to {fig_path}")
        
    except Exception as e:
        print(f"Error plotting PSD comparison: {str(e)}")

def analyze_frequency_bands(results):
    """Analyze frequency band powers and their correlations"""
    from scipy.stats import pearsonr
    
    print("\nAnalyzing frequency bands...")
    band_powers_real = results['band_powers_real']
    band_powers_imputed = results['band_powers_imputed']
    
    # Calculate correlations for each band
    correlations = {}
    for band in band_powers_real:
        corr = pearsonr(band_powers_real[band], band_powers_imputed[band])[0]
        correlations[band] = corr
    
    # Print correlations
    print("\nFrequency Band Power Correlations:")
    for band, corr in correlations.items():
        print(f"{band.capitalize()}: {corr:.3f}")
    
    # Create scatter plots for each band
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, band in enumerate(band_powers_real):
        ax = axes[idx]
        ax.scatter(band_powers_real[band], band_powers_imputed[band], 
                  alpha=0.5, label=f'r = {correlations[band]:.3f}')
        
        # Add diagonal line
        min_val = min(band_powers_real[band].min(), band_powers_imputed[band].min())
        max_val = max(band_powers_real[band].max(), band_powers_imputed[band].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        ax.set_xlabel('Real Power')
        ax.set_ylabel('Imputed Power')
        ax.set_title(f'{band.capitalize()} Band Power')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return correlations