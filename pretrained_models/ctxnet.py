"""
CtxNet model definition (same architecture as in
"1. CtxNet Implementation for Spectral Feature Modeling/3. CtxNet Implementation - Transfer Learning.ipynb").
"""
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft


def apply_hilbert_torch(x, envelope=False, do_log=False, compute_val='power', data_srate=250):
    def hilbert_torch(x):
        N = x.size(-1)

        # Disable mixed precision for FFT
        with torch.autocast(device_type=x.device.type, enabled=False):
            Xf = fft.fft(x.float(), dim=-1)

        h = torch.zeros(N, dtype=torch.complex64, device=x.device)
        if N % 2 == 0:
            h[0] = h[N // 2] = 1
            h[1:N // 2] = 2
        else:
            h[0] = 1
            h[1:(N + 1) // 2] = 2

        Xf_hilbert = Xf * h
        x_hilbert = fft.ifft(Xf_hilbert, dim=-1)

        return x_hilbert

    def angle_custom(z):
        return torch.atan2(z.imag, z.real)

    def unwrap(p, discont=np.pi):
        dp = p[..., 1:] - p[..., :-1]
        ddp = torch.remainder(dp + np.pi, 2 * np.pi) - np.pi
        ddp[torch.abs(dp) < discont] = 0
        p_unwrapped = p.clone()
        p_unwrapped[..., 1:] = p[..., 0][..., None] + torch.cumsum(dp + ddp, dim=-1)
        return p_unwrapped

    def diff(x):
        return x[..., 1:] - x[..., :-1]

    hilb_sig = hilbert_torch(x)

    if compute_val == 'power':
        out = torch.abs(hilb_sig)
        if do_log:
            out = torch.log1p(out)
    elif compute_val == 'phase':
        out = unwrap(angle_custom(hilb_sig))
    elif compute_val == 'freqslide':
        ang = angle_custom(hilb_sig)
        ang = data_srate * diff(unwrap(ang)) / (2 * np.pi)
        out = torch.nn.functional.pad(ang, (0, 1), mode='constant')
    return out


class CtxNet(nn.Module):
    def __init__(self, Chans=3, Samples=375, dropoutRate=0.65, kernLength=64, F1=4,
                 D=2, F2=8, F3=16, norm_rate=0.25, kernLength_sep=16,
                 do_log=False, data_srate=1, base_split=4):
        super(CtxNet, self).__init__()
        self.do_log = do_log
        self.data_srate = data_srate

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, (Chans, 1), groups=F1, bias=False, padding='same'),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F2, (1, kernLength_sep), bias=False, padding='same'),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(F2, F3, (1, kernLength_sep//2), bias=False, padding='same'),
            nn.BatchNorm2d(F3),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )

        self.flatten = nn.Flatten()

        flatten_size = self.calculate_flatten_size(Chans, Samples, F3)

        self.dense = nn.Sequential(
            nn.Linear(flatten_size, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(dropoutRate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(dropoutRate)
        )

        self.output = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )

    def calculate_flatten_size(self, Chans, Samples, F3):
        with torch.no_grad():
            x = torch.randn(1, 1, Chans, Samples)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            return x.numel()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.apply_hilbert(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.output(x)
        return x

    def apply_hilbert(self, x):
        return apply_hilbert_torch(x, do_log=self.do_log, compute_val='power', data_srate=self.data_srate)
