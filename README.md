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

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/64cb2285-bfc7-4920-92dc-4119c7af85a8" />

---

## References

This project builds on previous work in neural decoding and generative modeling:

- Peterson SM, Steine-Hanson Z, Davis N, Rao RPN, Brunton BW.  
  *Generalized neural decoders for transfer learning across participants and recording modalities.*  
  J Neural Eng. 2021;18(2). [https://doi.org/10.1088/1741-2552/abda0b](https://doi.org/10.1088/1741-2552/abda0b)

- Vetter J, Macke JH, Gao R.  
  *Generating realistic neurophysiological time series with denoising diffusion probabilistic models.*  
  Patterns. 2024;5(9). [https://doi.org/10.1016/j.patter.2024.101047](https://doi.org/10.1016/j.patter.2024.101047)

---

## Contact

For questions, suggestions, or collaborations, please contact **zixiao_yin@ccmu.edu.cn** or open an issue on this repository.
