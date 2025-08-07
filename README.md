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

## Contact

For questions, suggestions, or collaborations, please contact **zixiao_yin@ccmu.edu.cn** or open an issue on this repository.
