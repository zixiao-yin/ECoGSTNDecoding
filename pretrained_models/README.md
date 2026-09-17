# Pretrained base models

Two base models trained on chronic ECoG and STN-LFP recordings. They are intended as
starting points for transfer learning to new subjects, as described in the paper.

| File | Model |
|---|---|
| `ctxnet_base_model.pth` | CtxNet base model (spectral feature decoding) |
| `ddpm_base_model.pt` + `ddpm_base_model.yaml` | DDPM base model (raw signal reconstruction) and its configuration |
| `ctxnet.py` | CtxNet model definition |
| `load_pretrained.py` | Functions to load both models |

## CtxNet base model

- Input: 3 ECoG channels, 1.5-s windows at 250 Hz, shape `(n, 3, 375)`, each window z-scored over time.
- Output: one value per window, the predicted STN high-beta power (standardized units).
- Used by `1. CtxNet Implementation for Spectral Feature Modeling/3. CtxNet Implementation - Transfer Learning.ipynb`.
  Point the `torch.load(...)` path in that notebook to `pretrained_models/ctxnet_base_model.pth`.

## DDPM base model

- Input: 3 ECoG channels, 4-s windows at 250 Hz, shape `(n, 3, 1000)`, each window and channel z-scored.
- Output: a generated STN-LFP window, shape `(n, 1, 1000)`, in z-scored units.
- Generation is stochastic and uses the full 1000-step denoising process.
- Requires the `ntd` package in `2. DDPM Implementation for Raw Signal Reconstruction/`.

## Usage

Keep this folder at the top level of the repository, then:

```python
import sys
sys.path.insert(0, "pretrained_models")
from load_pretrained import load_ctxnet_base, load_ddpm_base, ddpm_generate, zscore_windows

ctxnet = load_ctxnet_base()
ddpm = load_ddpm_base()

# ecog: numpy array (n, 3, 1000), 250 Hz
stn_generated = ddpm_generate(ddpm, zscore_windows(ecog))
```

`python pretrained_models/load_pretrained.py` checks that both models load.

Please refer to the paper for data, preprocessing and evaluation details, and cite it when using these models.
