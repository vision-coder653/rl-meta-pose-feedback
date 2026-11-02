"""UmeTrack's original skinned surface, in the same metres/side as human FK.

Model and skinning: Meta Platforms, Inc. and affiliates, CC-BY-NC-SA-4.0.
See core/resources/umetrack/UPSTREAM.md and LICENSE under the experiment directory.
This adapter does not retarget.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .human import UmeTrackFK
from .umetrack.hand_skinning import _skin_points
from emg_control_paths import RESOURCES_DIR


class UmeTrackMesh(nn.Module):
    def __init__(self, *, side: str, mesh_path: Path | None = None):
        super().__init__()
        directory = RESOURCES_DIR / "umetrack"
        profile = json.loads((directory / "generic_hand_model.json").read_text())
        path = directory / "generic_hand_mesh.json" if mesh_path is None else Path(mesh_path)
        mesh = json.loads(path.read_text(encoding="utf-8"))
        fingerprint = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if mesh["provenance"]["fk_profile_sha256"] != fingerprint:
            raise ValueError("Human mesh profile differs from the FK geometry used for reward")
        self.fk = UmeTrackFK(side=side)
        vertices = torch.tensor(mesh["mesh_vertices"], dtype=torch.float64) * .001
        weights = torch.tensor(mesh["dense_bone_weights"], dtype=torch.float64)
        faces = torch.tensor(mesh["mesh_triangles"], dtype=torch.long)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or weights.shape != (len(vertices), 17)
                or faces.ndim != 2 or faces.shape[1] != 3 or not torch.isfinite(vertices).all()
                or not torch.isfinite(weights).all() or weights.min() < 0
                or not torch.allclose(weights.sum(-1), torch.ones(len(vertices), dtype=weights.dtype), atol=1e-5)
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("Invalid UmeTrack surface or skinning weights")
        if side == "right":
            faces = faces.flip(-1)  # Reflection reverses surface winding once.
        self.register_buffer("rest_vertices", vertices)
        self.register_buffer("weights", weights)
        self.register_buffer("faces", faces)
        self.provenance = mesh["provenance"]

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if (angles.ndim < 1 or angles.shape[-1] != 20 or not angles.is_floating_point()
                or not torch.isfinite(angles).all()):
            raise ValueError("Human mesh angles must be finite floating [...,20] radians")
        shape, flat = angles.shape[:-1], angles.reshape(-1, 20)
        count = len(flat)
        if not count:
            return angles.new_empty((*shape, len(self.rest_vertices), 3))
        expand = lambda value: value.to(angles).expand(count, *value.shape)
        wrist = torch.eye(4, dtype=angles.dtype, device=angles.device).repeat(count, 1, 1)
        if self.fk.side == "right":
            wrist[:, 0, 0] = -1
        vertices = _skin_points(expand(self.fk.joint_rest_positions), expand(self.fk.joint_rotation_axes),
                                expand(self.weights), flat, expand(self.rest_vertices), wrist)
        return vertices.reshape(*shape, len(self.rest_vertices), 3)
