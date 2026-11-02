"""Group-shared trainable 5 x 20 linear mapping, followed by a motor mask."""
from __future__ import annotations

import torch
from torch import nn

from .command_head import expand_condition


class Synergy(nn.Module):
    def __init__(self, initial_matrix: torch.Tensor):
        super().__init__()
        if (initial_matrix.shape != (5, 20) or not initial_matrix.is_floating_point()
                or not torch.isfinite(initial_matrix).all()):
            raise ValueError("initial_matrix must be a finite floating point tensor [5,20]")
        # Initialization is explicit: the component chooses no scientific prior.
        self.B = nn.Parameter(initial_matrix.detach().clone())

    @classmethod
    def orthogonal(cls):
        """Approved baseline initializer; orthogonality is not enforced later."""
        matrix = torch.empty(5, 20)
        nn.init.orthogonal_(matrix)
        return cls(matrix)

    def forward(self, coefficients: torch.Tensor, motor_mask: torch.Tensor) -> torch.Tensor:
        if coefficients.ndim not in (2, 3) or coefficients.shape[-1] != 5:
            raise ValueError("coefficients must be [B,5] or [B,T,5]")
        mask = expand_condition(motor_mask, coefficients, 20, "motor_mask")
        if mask.dtype != torch.bool:
            raise ValueError("motor_mask must be boolean")
        return (coefficients @ self.B).masked_fill(~mask, 0)
