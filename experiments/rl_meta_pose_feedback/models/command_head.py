"""Fuse EMG hidden, morphology and the masked internally issued robot target."""
from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn


def expand_condition(values: torch.Tensor, reference: torch.Tensor, width: int,
                     name: str) -> torch.Tensor:
    """Broadcast one condition globally, per batch item, or per token."""
    if values.shape[-1:] != (width,):
        raise ValueError(f"{name} must have final dimension {width}")
    if values.device != reference.device:
        raise ValueError(f"{name} must be on the same device as the latent")
    if values.ndim == 1:
        return values.expand(*reference.shape[:-1], width)
    if reference.ndim == 3 and values.ndim == 2 and values.shape[0] == reference.shape[0]:
        return values[:, None].expand(-1, reference.shape[1], -1)
    if values.shape[:-1] != reference.shape[:-1]:
        raise ValueError(f"{name} has incompatible batch/time dimensions")
    return values


class CommandHead(nn.Module):
    def __init__(self, latent_dim: int, morphology_dim: int, *,
                 hidden_sizes: Sequence[int], activation: str):
        super().__init__()
        if (type(latent_dim) is not int or latent_dim <= 0
                or type(morphology_dim) is not int or morphology_dim <= 0
                or any(type(size) is not int or size <= 0 for size in hidden_sizes)):
            raise ValueError("Input and hidden dimensions must be positive integers")
        activations = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        if activation not in activations:
            raise ValueError(f"activation must be one of {tuple(activations)}")
        self.latent_dim, self.morphology_dim = latent_dim, morphology_dim
        self.hidden_sizes, self.activation = tuple(hidden_sizes), activation
        # Keep absolute angle magnitudes: do not LayerNorm the 20-angle inputs.
        self.latent_projection = nn.Sequential(nn.Linear(latent_dim, 128), nn.LayerNorm(128), nn.GELU())
        self.command_projection = nn.Sequential(nn.Linear(20, 64), nn.GELU())
        widths = (128 + 64 + morphology_dim, *self.hidden_sizes)
        layers = []
        for input_width, output_width in zip(widths[:-1], widths[1:]):
            layers.extend((nn.Linear(input_width, output_width), activations[activation]()))
        layers.append(nn.Linear(widths[-1], 5))
        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, morphology: torch.Tensor, *,
                command_pose: torch.Tensor, motor_mask: torch.Tensor) -> torch.Tensor:
        if latent.ndim not in (2, 3) or latent.shape[-1] != self.latent_dim:
            raise ValueError("latent must be [B,D] or [B,T,D] with the configured latent width")
        morphology = expand_condition(morphology, latent, self.morphology_dim, "morphology")
        if not morphology.is_floating_point() or not torch.isfinite(morphology).all():
            raise ValueError("morphology must contain finite floating point features")
        mask = expand_condition(motor_mask, latent, 20, "motor_mask")
        if mask.dtype != torch.bool:
            raise ValueError("motor_mask must be boolean")
        if (command_pose.shape != (*latent.shape[:-1], 20) or command_pose.device != latent.device
                or not command_pose.is_floating_point()):
            raise ValueError("command_pose must be aligned 20-angle inputs")
        # Missing slots have no angle. Sanitize before finite checks or scaling;
        # morphology's presence bits distinguish them from real zero angles.
        command_pose = command_pose.masked_fill(~mask, 0)
        if not torch.isfinite(command_pose).all():
            raise ValueError("Active command angles must be finite")
        fused = torch.cat((self.latent_projection(latent),
                           self.command_projection(command_pose.to(latent.dtype) / math.pi),
                           morphology), dim=-1)
        return self.network(fused)
