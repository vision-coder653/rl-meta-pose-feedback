"""Meta FK adapter and nearest-sample reference lookup within one valid segment.

The static generic hand describes a left hand. Right-hand geometry is reflected
once at the wrist transform. Output is wrist-local, in metres, not mocap world
positions or participant-specific geometry. See core/resources/umetrack/UPSTREAM.md
and LICENSE under the experiment directory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from torch import nn

from .umetrack.hand import HandModel
from .umetrack.hand_skinning import skin_landmarks
from .umetrack.hand_skinning import _hand_skinning_transform
from .frames import finger_frame
from emg_control_paths import RESOURCES_DIR

WRIST_INDEX = 5
FINGERTIP_INDICES = (0, 1, 2, 3, 4)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


class UmeTrackFK(nn.Module):
    """Angles [..., 20] in rad -> landmarks [..., 21, 3] in metres.

    Landmark order is UmeTrack order: five tips, wrist, finger frames, palm.
    The EMG2Pose public wrapper's [B, 20, T] layout is intentionally not used.
    """

    def __init__(self, *, side: str):
        super().__init__()
        if side not in ("left", "right"):
            raise ValueError("side must explicitly be 'left' or 'right'")
        self.side = side
        profile = json.loads((RESOURCES_DIR / "umetrack/generic_hand_model.json").read_text())
        integer_fields = {"joint_frame_index", "joint_parent", "joint_first_child", "joint_next_sibling",
                          "landmark_rest_bone_indices"}
        # Preserve model precision when callers request double-precision FK.
        for name, value in profile.items():
            tensor = torch.tensor(value, dtype=torch.long if name in integer_fields else torch.float64)
            if name in ("joint_rest_positions", "landmark_rest_positions"):
                tensor = tensor * 0.001
            self.register_buffer(name, tensor)
        # Rest frames are attached to distal bones. Their origin will be the tip
        # landmark; rotations are transported by that bone's rigid FK transform.
        rest_frames = []
        for finger in range(5):
            joint = 4 * finger + 3
            direction = self.landmark_rest_positions[finger] - self.joint_rest_positions[joint]
            rest_frames.append(torch.from_numpy(finger_frame(direction.numpy(),
                                                             self.joint_rotation_axes[joint].numpy())))
        self.register_buffer("distal_rest_frames", torch.stack(rest_frames))

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if (not isinstance(angles, torch.Tensor) or angles.ndim < 1 or angles.shape[-1] != 20
                or not angles.is_floating_point() or not torch.isfinite(angles).all()):
            raise ValueError("angles must be a finite floating tensor [..., 20], in radians")
        shape = angles.shape[:-1]
        flat = angles.reshape(-1, 20)
        count = flat.shape[0]
        if not count:
            return angles.new_empty((*shape, 21, 3))
        fields = {}
        for name in HandModel._fields:
            if not hasattr(self, name):
                continue
            value = getattr(self, name)
            dtype = angles.dtype if value.is_floating_point() else value.dtype
            fields[name] = value.to(device=angles.device, dtype=dtype).expand(count, *value.shape)
        model = HandModel(**fields)
        wrist = torch.eye(4, dtype=angles.dtype, device=angles.device).repeat(count, 1, 1)
        if self.side == "right":
            wrist[:, 0, 0] = -1
        return skin_landmarks(model, flat, wrist).reshape(*shape, 21, 3)

    def fingertip_frames(self, angles: torch.Tensor) -> torch.Tensor:
        """Return [..., 5, 3, 3] proper rotations from finger-local to wrist axes.

        +X follows the distal direction projected perpendicular to the positive
        flexion hinge (+Y); +Z completes a right-handed frame.
        Right-hand reflection acts differently on a direction and an angular
        axis; copying the reflected skinning matrix would have determinant -1.
        """
        if (not isinstance(angles, torch.Tensor) or angles.ndim < 1 or angles.shape[-1] != 20
                or not angles.is_floating_point() or not torch.isfinite(angles).all()):
            raise ValueError("angles must be a finite floating tensor [..., 20], in radians")
        shape = angles.shape[:-1]
        flat = angles.reshape(-1, 20)
        count = flat.shape[0]
        if not count:
            return angles.new_empty((*shape, 5, 3, 3))
        identity = torch.eye(4, device=angles.device, dtype=angles.dtype).expand(count, 4, 4)
        transforms = _hand_skinning_transform(
            self.joint_rotation_axes[:20].to(angles).expand(count, 20, 3),
            self.joint_rest_positions[:20].to(angles).expand(count, 20, 3), flat, identity)
        distal_indices = torch.tensor([4, 7, 10, 13, 16], device=angles.device)
        rotation = transforms[:, distal_indices, :3, :3] @ self.distal_rest_frames.to(angles)
        if self.side == "right":
            # Polar directions reflect by M; positive angular axes by det(M) M.
            rotation = (angles.new_tensor([-1., 1., 1.])[:, None] * rotation
                        * angles.new_tensor([1., -1., 1.]))
        return rotation.reshape(*shape, 5, 3, 3)

    def palm_frame(self, *, reference_finger: str) -> torch.Tensor:
        """Fixed palm-local -> wrist basis, calibrated from zero-pose geometry.

        Copy the reference finger's distal basis at zero angles, then freeze it
        in the wrist. +X is its distal direction, +Y positive flexion, +Z X cross
        Y. The palm origin is selected separately; its depth cannot tilt this
        basis. Groups use the corresponding middle finger on both sides.
        """
        if reference_finger not in FINGER_NAMES:
            raise ValueError("Palm frame needs a named reference finger")
        finger = FINGER_NAMES.index(reference_finger)
        zero = self.joint_rest_positions.new_zeros(20)
        return self.fingertip_frames(zero)[finger].detach().clone()


@dataclass(frozen=True)
class ReferenceSample:
    angles: torch.Tensor
    time: torch.Tensor
    indices: torch.Tensor
    query_time: torch.Tensor


class HumanReference:
    """One continuous, valid reference segment; no interpolation or extrapolation.

    max_gap_seconds is an explicit continuity rule, not a resampling interval.
    Query tolerance is also explicit and does not authorize crossing a gap.
    """

    def __init__(self, time: torch.Tensor, angles: torch.Tensor, *, max_gap_seconds: float):
        time = torch.as_tensor(time, dtype=torch.float64, device="cpu").clone()
        angles = torch.as_tensor(angles)
        if (time.ndim != 1 or not time.numel() or angles.shape != (time.numel(), 20)
                or not angles.is_floating_point() or not torch.isfinite(time).all()
                or not torch.isfinite(angles).all()):
            raise ValueError("Reference needs finite time [T] and radian angles [T, 20]")
        if not 0 < max_gap_seconds < float("inf"):
            raise ValueError("max_gap_seconds must be finite and positive")
        gaps = time[1:] - time[:-1]
        if torch.any(gaps <= 0) or torch.any(gaps > max_gap_seconds):
            raise ValueError("Reference crosses a time discontinuity; supply one valid segment")
        self.time, self.angles = time, angles.clone()

    def sample(self, query_time: torch.Tensor | float, *, max_error_seconds: float) -> ReferenceSample:
        if not 0 <= max_error_seconds < float("inf"):
            raise ValueError("max_error_seconds must be finite and nonnegative")
        query = torch.as_tensor(query_time, dtype=torch.float64, device="cpu")
        if (not torch.isfinite(query).all() or torch.any(query < self.time[0])
                or torch.any(query > self.time[-1])):
            raise ValueError("Reference query is outside the segment; no extrapolation")
        right = torch.searchsorted(self.time, query.contiguous()).clamp(max=len(self.time) - 1)
        left = (right - 1).clamp(min=0)
        # Ties choose the earlier sample, consistently for all candidate hands.
        indices = torch.where((query - self.time[left]).abs() <= (self.time[right] - query).abs(),
                              left, right)
        selected_time = self.time[indices]
        if torch.any((selected_time - query).abs() > max_error_seconds):
            raise ValueError("No reference sample within the explicit time tolerance")
        return ReferenceSample(self.angles[indices.to(self.angles.device)], selected_time,
                               indices, query.clone())
