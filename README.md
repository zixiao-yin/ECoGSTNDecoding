# ECoGSTNDecoding

**ECoG-based STN Decoding using Deep Learning and Diffusion Models**

## Overview

Next-generation neurotechnologies aim to decode brain activity to guide assistive brain-computer interfaces (BCIs) and closed-loop neuromodulation therapies. These applications often rely on sensing electrophysiological signals from deep brain targets (e.g., subthalamic nucleus, globus pallidus internus, thalamus), which are small and deeply located structures associated with distinct disease signatures.

However, extracting meaningful information from these targets is challenging due to:
- Low signal-to-noise ratio (SNR)
- Stimulation and physiological artifacts
- Recording failures or incomplete data

To address these limitations, we propose a hybrid decoding framework that leverages **cortical electrocorticography (ECoG)** signals to reconstruct deep brain activity using:
- **CtxNet** – a deep spectral model for decoding subcortical activity
- **DDPM** – a generative diffusion model for raw signal reconstruction

Our approach has been validated on **723 hours of recordings from 49 patients**, spanning:
- Multiple diseases: Parkinson’s disease, dystonia, Tourette syndrome  
- Behavioral states: rest, movement, sleep  
- Deep brain targets: STN, GPi, thalamus

---

## Key Contributions

- 💡 **CtxNet**: Architecture for modeling spectral features of cortical signals
- 🌀 **DDPM**: Denoising Diffusion Probabilistic Model for signal imputation
- 🧠 **Clinical utility**: Robust decoding during DBS lead failures for sleep staging and movement detection
- 🔮 **Future-ready**: Enabling deep brain state prediction without relying on invasive DBS signals

---

## Visual Summary

![Model Overview](https://github.com/user-attachments/assets/81a0f0a4-c51e-4220-b8b6-bb1fcb7b9c45)

---

## System Requirements

### Software Dependencies
This software has been tested on the following system configuration:

- **Python**: 3.10.18 or higher
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS 10.15+
- **CUDA**: 12.6 or higher (for GPU acceleration)

### Required Python Packages
```
torch==2.8.0+cu126 (or compatible PyTorch version with CUDA support)
numpy==2.1.2
scipy==1.15.3
pandas==2.3.1
matplotlib==3.10.5
scikit-learn==1.7.1
mne==1.10.1
jupyter==1.1.1
notebook==7.4.5
ipython==7.34.0
tqdm==4.67.1
pillow==11.0.0
```

### Hardware Requirements
**Minimum Requirements:**
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 16 GB
- GPU: NVIDIA GPU with CUDA support (8+ GB VRAM recommended)
- Storage: 10 GB free disk space

**Recommended Configuration (used for testing):**
- CPU: 48 physical cores (96 logical cores)
- RAM: 128 GB
- GPU: NVIDIA GPU with CUDA 12.6+ (tested with single GPU)
- Storage: 50+ GB for large datasets

### Operating Systems Tested
- Windows 10/11 (AMD64)

---
## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/zixiao-yin/ECoGSTNDecoding.git
cd ECoGSTNDecoding
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Using conda
conda create -n ecog_stn python=3.10
conda activate ecog_stn

# Or using venv
python -m venv ecog_stn_env
source ecog_stn_env/bin/activate  # On Windows: ecog_stn_env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
# Visit https://pytorch.org/get-started/locally/ for the appropriate command
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
pip install numpy==2.1.2 scipy==1.15.3 pandas==2.3.1 matplotlib==3.10.5
pip install scikit-learn==1.7.1 mne==1.10.1 tqdm==4.67.1 pillow==11.0.0
pip install jupyter notebook ipython

# Or install all dependencies from requirements file
pip install -r requirements.txt
```
---
## Demo

### Quick Start with Demo Data
We provide a small demonstration dataset to test the software installation and basic functionality.

#### Step 1: Download Demo Data
```bash
# Demo data will be made available at [provide link or instructions]
# For now, use the example in the notebooks
```

#### Step 2: Run Demo Notebooks
```bash
# Launch Jupyter Notebook
jupyter notebook

# Navigate to and open:
# 1. "CtxNet Implementation - Architecture & Training & Validation Pipeline.ipynb"
# 2. "DDPM Implementation - Raw Signal Reconstruction.ipynb"
```

#### Expected Output
- **CtxNet Demo**: Should produce spectral decoding results with correlation metrics and visualization plots
- **DDPM Demo**: Should generate reconstructed waveforms and comparison with ground truth

#### Expected Runtime
- **CtxNet Demo**: ~2-5 minutes on a GPU, ~15-30 minutes on CPU
- **DDPM Demo**: ~5-10 minutes on a GPU, ~30-60 minutes on CPU

---

## Instructions for Use

### Running the Software on Your Own Data

#### Data Format Requirements
The software expects input data in the following format:
- **ECoG signals**: NumPy array format (`.npy`) or MNE-Python Raw objects
- **Shape**: `(n_channels, n_timepoints)` or `(n_epochs, n_channels, n_timepoints)`
- **Sampling rate**: 50 Hz or higher (will be resampled if needed)
- **Deep brain signals** (optional, for training): Same format as ECoG

#### Step 1: Prepare Your Data
```python
import numpy as np
import mne

# Load your ECoG data
ecog_data = np.load('your_ecog_data.npy')  # Shape: (n_channels, n_timepoints)
stn_data = np.load('your_stn_data.npy')    # Shape: (n_channels, n_timepoints)

# Or use MNE format
raw_ecog = mne.io.read_raw_fif('your_ecog_data.fif')
```

#### Step 2: CtxNet Training
Open and follow the notebook:
```
"1. CtxNet Implementation for Spectral Feature Modeling/CtxNet Implementation - Architecture & Training & Validation Pipeline.ipynb"
```

Key steps:
1. Load your ECoG and STN data
2. Configure model hyperparameters
3. Train the CtxNet model
4. Evaluate decoding performance
5. Save trained model

#### Step 3: DDPM Signal Reconstruction
Open and follow the notebook:
```
"2. DDPM Implementation for Raw Signal Reconstruction/DDPM Implementation - Training Base Model.ipynb"
```

Key steps:
1. Load preprocessed data
2. Configure diffusion model parameters
3. Train the DDPM model
4. Generate reconstructed signals
5. Evaluate reconstruction quality

#### Step 4: Transfer Learning (Optional)
For adapting models to new subjects or conditions:
- CtxNet: `CtxNet Implementation - Transfer Learning.ipynb`
- DDPM: `DDPM Implementation - Transfer Learning.ipynb`

### Expected Outputs
- **CtxNet**: Decoded spectral features, correlation metrics, prediction accuracy
- **DDPM**: Reconstructed waveforms, signal quality metrics, visualization plots

---

## Repository Structure

## ECoGSTNDecoding/

### 1. CtxNet Implementation for Spectral Feature Modeling/
- CtxNet Implementation - Architecture & Training & Validation Pipeline.ipynb
- CtxNet Implementation - Learning Curve.ipynb
- CtxNet Implementation - Transfer Learning.ipynb

### 2. DDPM Implementation for Raw Signal Reconstruction/
- DDPM Implementation - Raw Signal Reconstruction.ipynb
- DDPM Implementation - Batch Processing.ipynb
- DDPM Implementation - Learning Curve.ipynb
- DDPM Implementation - Training Base Model.ipynb
- DDPM Implementation - Transfer Learning.ipynb

### Diffusion_STN_Generator/
- (Early experimental tests by Zixuan Liu; includes `npy_data_reso50hz`)

---
## Reproduction Instructions

To reproduce the results presented in the manuscript:

1. **Data Availability**: The full clinical dataset used in this study requires ethics approval and data use agreements. Interested researchers should contact the corresponding author.

2. **Model Training**: Follow the notebooks in sequential order:
   - Train CtxNet using the full dataset
   - Train DDPM using the full dataset
   - Evaluate on held-out test sets

3. **Performance Metrics**: All performance metrics reported in the manuscript can be computed using the evaluation functions provided in the notebooks.

4. **Computational Requirements**: Full training requires GPU resources. Training time: ~48-72 hours for CtxNet, ~72-120 hours for DDPM on the full dataset.

---

## Acknowledgments

This work was inspired by and builds upon methods from:
- **Neural Timeseries Diffusion**: https://github.com/mackelab/neural_timeseries_diffusion
- **HTNet Generalized Decoding**: https://github.com/BruntonUWBio/HTNet_generalized_decoding

We thank the authors of these repositories for making their code publicly available.

---
## Contact

For questions, suggestions, or collaborations, please contact **zixiao_yin@ccmu.edu.cn** or open an issue on this repository.
