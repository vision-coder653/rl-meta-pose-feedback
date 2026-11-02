"""Independent critic and explicit raw/train-standardized EMG input handling."""
from __future__ import annotations

import math

import torch
from torch import nn


class EMGNormalizer(nn.Module):
    def __init__(self, *, clip: float, mode: str = "standardize"):
        super().__init__()
        if not math.isfinite(clip) or clip <= 0:
            raise ValueError("Normalization clip must be finite and positive")
        self.clip = float(clip)
        if mode not in ("raw", "standardize"):
            raise ValueError("EMG preprocessing mode must be raw or standardize")
        self.mode = mode
        self.register_buffer("mean", torch.zeros(16))
        self.register_buffer("std", torch.ones(16))
        self.register_buffer("fitted", torch.tensor(mode == "raw"))

    @torch.no_grad()
    def fit(self, windows):
        if self.mode == "raw":
            raise RuntimeError("Raw EMG input does not fit or apply normalization statistics")
        if self.fitted:
            raise RuntimeError("Normalization is frozen after fitting on training windows")
        count, mean, m2 = 0, torch.zeros(16, dtype=torch.float64), torch.zeros(16, dtype=torch.float64)
        for window in windows:
            x = torch.as_tensor(window, dtype=torch.float64, device="cpu")
            if x.ndim != 2 or x.shape[-1] != 16 or not x.shape[0] or not torch.isfinite(x).all():
                raise ValueError("Fit needs finite training EMG [samples,16]")
            n = x.shape[0]
            delta = x.mean(0) - mean
            m2 += x.var(0, unbiased=False) * n + delta.square() * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
        if count < 2:
            raise ValueError("Need training samples to fit normalization")
        self.mean.copy_(mean.to(self.mean))
        self.std.copy_((m2 / count).sqrt().clamp_min(1e-6).to(self.std))
        self.fitted.fill_(True)

    def forward(self, emg: torch.Tensor):
        if not self.fitted or emg.shape[-1] != 16 or not torch.isfinite(emg).all():
            raise ValueError("Use fitted normalization and finite 16-channel EMG")
        if self.mode == "raw":
            return emg
        return ((emg - self.mean) / self.std).clamp(-self.clip, self.clip)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        self.input_dim = input_dim
        layers = []
        widths = (input_dim, *hidden_sizes, 1)
        for i, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            linear = nn.Linear(left, right)
            nn.init.orthogonal_(linear.weight, gain=1. if right == 1 else math.sqrt(2))
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < len(widths)-2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor):
        if observations.shape[-1] != self.input_dim or not torch.isfinite(observations).all():
            raise ValueError("Critic observations have wrong width or nonfinite values")
        return self.network(observations).squeeze(-1)
