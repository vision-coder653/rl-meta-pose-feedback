"""Independent Gaussian motor exploration after the trainable synergy map.

Actions are raw motor-angle increments in radians, before environment clipping.
Only active motors contribute to likelihood and entropy; padded tokens have
zero actions, log probability and entropy. This module implements the policy
distribution, not a reward, critic or training algorithm.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Normal

from .command_head import expand_condition


class MotorNoise(nn.Module):
    """One global log standard deviation per motor, shared across inputs.

    The initial standard deviation is mandatory, in radians per command.
    Learning it is the selected baseline; a fixed-buffer option is available
    for comparisons. No initial exploration amplitude is selected here.
    """

    def __init__(self, initial_std: float | Sequence[float] | torch.Tensor, *,
                 learnable: bool = True):
        super().__init__()
        if type(learnable) is not bool:
            raise ValueError("learnable must be boolean")
        initial = torch.as_tensor(initial_std)
        if initial.dtype == torch.bool or initial.is_complex():
            raise ValueError("initial_std must contain positive finite real values")
        initial = initial.detach().to(dtype=torch.float32).clone()
        if initial.ndim == 0:
            initial = initial.expand(20).clone()
        if initial.shape != (20,) or not torch.isfinite(initial).all() or torch.any(initial <= 0):
            raise ValueError("initial_std must be a positive finite scalar or 20-slot vector in radians")
        if learnable:
            self.log_std = nn.Parameter(initial.log())
        else:
            self.register_buffer("log_std", initial.log())

    @property
    def stddev(self) -> torch.Tensor:
        return self.log_std.exp()


class MaskedMotorNormal:
    """Gaussian policy over active entries of [B,20] or [B,T,20] means.

    std accepts a scalar, [20], [B,20] or a tensor matching mean. A tensor std
    retains its gradient, including one produced by MotorNoise.stddev.
    Re-score stored, unclipped actions with log_prob(); action values are
    detached internally so policy gradients only follow distribution parameters.
    """

    def __init__(self, mean: torch.Tensor, motor_mask: torch.Tensor, *,
                 std: float | torch.Tensor):
        if (mean.ndim not in (2, 3) or mean.shape[-1] != 20
                or not mean.is_floating_point()):
            raise ValueError("mean must be floating point [B,20] or [B,T,20]")
        mask = expand_condition(motor_mask, mean, 20, "motor_mask")
        if mask.dtype != torch.bool:
            raise ValueError("motor_mask must be boolean")
        if isinstance(std, torch.Tensor):
            if std.device != mean.device:
                raise ValueError("std must be on the same device as mean")
            if not std.is_floating_point():
                raise ValueError("std tensor must be floating point")
            scale = std.to(dtype=mean.dtype)
        else:
            if isinstance(std, bool) or not isinstance(std, (int, float)):
                raise ValueError("std must be a positive scalar or floating point tensor")
            scale = mean.new_tensor(std)
        scale = (scale.expand_as(mean) if scale.ndim == 0 else
                 expand_condition(scale, mean, 20, "std"))
        self.motor_mask = mask.clone()
        # Use neutral parameters before doing probability arithmetic on absent
        # motors; multiplying NaN/inf log probabilities by zero would not work.
        self.mean = mean.masked_fill(~self.motor_mask, 0)
        scale = scale.masked_fill(~self.motor_mask, 1)
        if (not torch.isfinite(self.mean).all() or not torch.isfinite(scale).all()
                or torch.any(scale <= 0)):
            raise ValueError("Active motor means must be finite and std must be positive and finite")
        self._normal = Normal(self.mean, scale)

    @property
    def stddev(self) -> torch.Tensor:
        return self._normal.scale.masked_fill(~self.motor_mask, 0)

    def sample(self) -> torch.Tensor:
        """Sample detached raw increments; the environment clips them later."""
        return self._normal.sample().masked_fill(~self.motor_mask, 0)

    def log_prob(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Joint log density over active motors, before clipping; shape [B,T]/[B]."""
        if (raw_action.shape != self.mean.shape or raw_action.device != self.mean.device
                or not raw_action.is_floating_point()):
            raise ValueError("raw_action must be floating point with the mean's shape and device")
        action = raw_action.detach().masked_fill(~self.motor_mask, 0)
        if not torch.isfinite(action).all():
            raise ValueError("Active raw actions must be finite")
        result = self._normal.log_prob(action).masked_fill(~self.motor_mask, 0).sum(dim=-1)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite Gaussian log probability; check action and noise scale")
        return result

    def entropy(self) -> torch.Tensor:
        """Sum differential entropy over active motors, with padded tokens zero."""
        return self._normal.entropy().masked_fill(~self.motor_mask, 0).sum(dim=-1)
