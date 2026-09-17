"""
Load the pretrained base models.

    from load_pretrained import load_ddpm_base, load_ctxnet_base, zscore_windows

Run this file directly for a quick check that both models load:

    python load_pretrained.py
"""
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# the ntd package lives in the DDPM folder of this repository
sys.path.insert(0, os.path.join(REPO, "2. DDPM Implementation for Raw Signal Reconstruction"))
sys.path.insert(0, HERE)

DDPM_WEIGHTS = os.path.join(HERE, "ddpm_base_model.pt")
DDPM_CONFIG = os.path.join(HERE, "ddpm_base_model.yaml")
CTXNET_WEIGHTS = os.path.join(HERE, "ctxnet_base_model.pth")


def zscore_windows(x, eps=1e-8):
    """Z-score each window and channel over time. x: (n_windows, n_channels, n_samples)."""
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def load_ddpm_base(device=None):
    """Return the DDPM base model (ntd Diffusion) with pretrained weights, in eval mode."""
    from ntd.diffusion_model import Diffusion
    from ntd.networks import AdaConv
    from ntd.utils.kernels_and_diffusion_utils import OUProcess

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with open(DDPM_CONFIG) as f:
        cfg = yaml.safe_load(f)
    net, dif, ker = cfg["network"], cfg["diffusion"], cfg["diffusion_kernel"]
    length = cfg["dataset"]["signal_length"]

    network = AdaConv(
        signal_length=length,
        signal_channel=net["signal_channel"],
        cond_dim=net["cond_dim"],
        hidden_channel=net["hidden_channel"],
        in_kernel_size=net["in_kernel_size"],
        out_kernel_size=net["out_kernel_size"],
        slconv_kernel_size=net["slconv_kernel_size"],
        num_scales=net["num_scales"],
        num_blocks=net["num_blocks"],
        num_off_diag=net["num_off_diag"],
        use_pos_emb=net["use_pos_emb"],
        padding_mode=net["padding_mode"],
        use_fft_conv=net["use_fft_conv"],
    ).to(device)
    ou_process = OUProcess(ker["sigma_squared"], ker["ell"], length).to(device)
    model = Diffusion(
        network=network,
        noise_sampler=ou_process,
        mal_dist_computer=ou_process,
        diffusion_time_steps=dif["diffusion_steps"],
        schedule=dif["schedule"],
        start_beta=dif["start_beta"],
        end_beta=dif["end_beta"],
    ).to(device)
    model.load_state_dict(torch.load(DDPM_WEIGHTS, map_location=device, weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def ddpm_generate(model, ecog_windows, batch_size=512):
    """
    ecog_windows: (n, 3, 1000) array, already z-scored per window (see zscore_windows).
    Returns generated STN-LFP windows, (n, 1, 1000), in z-scored units.
    """
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(ecog_windows), batch_size):
        cond = torch.as_tensor(ecog_windows[i:i + batch_size], dtype=torch.float32, device=device)
        out.append(model.sample(num_samples=cond.shape[0], cond=cond, noise_type="alpha_beta").cpu().numpy())
    return np.concatenate(out, axis=0)


def load_ctxnet_base(device=None):
    """Return the CtxNet base model with pretrained weights, in eval mode."""
    from ctxnet import CtxNet

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(CTXNET_WEIGHTS, map_location=device, weights_only=True)
    model = CtxNet(**ckpt["model_params"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    ctx = load_ctxnet_base(device)
    x = zscore_windows(rng.standard_normal((8, 3, 375)))
    with torch.no_grad():
        y = ctx(torch.as_tensor(x, device=device))
    print("CtxNet loaded. input", x.shape, "-> output", tuple(y.shape))

    ddpm = load_ddpm_base(device)
    x = zscore_windows(rng.standard_normal((2, 3, 1000)))
    y = ddpm_generate(ddpm, x)
    print("DDPM loaded. input", x.shape, "-> output", y.shape)
