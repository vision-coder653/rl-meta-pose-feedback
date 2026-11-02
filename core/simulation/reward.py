"""Group-specific DexPilot-style vector error and actual-state reward.

The geometric metric is negative mean squared vector error (factor 1/2).
The shared open-pose cost is added by retargeting and aligned PPO rollouts.
Each vector is expressed in its origin frame, as in DexPilot VII-A. Fingertip
pairs are bidirectional; palm vectors point from the fixed palm point to tips.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from simulation.environment import HandState
from configs.groups import HUMAN_FINGERS, TrainingGroupConfig
from kinematics import HumanReference, RobotLandmarks, UmeTrackFK
from kinematics.human import WRIST_INDEX


def open_pose_cost(joint_angles, neutral, movable, coefficient):
    """Per-pose prior over movable joints; accepts [..., 20] actual angles.

    Neutral is the legal zero/open pose. Locked and inactive slots do not
    contribute to either the numerator or the denominator.
    """
    angles = np.asarray(joint_angles, dtype=np.float64)
    neutral = np.asarray(neutral, dtype=np.float64)
    movable = np.asarray(movable)
    if (angles.ndim < 1 or angles.shape[-1] != 20 or neutral.shape != (20,)
            or movable.shape != (20,) or movable.dtype != np.bool_
            or not np.isfinite(angles).all() or not np.isfinite(neutral).all()
            or not np.isfinite(coefficient) or coefficient < 0):
        raise ValueError("Open-pose cost needs finite 20-slot angles, a boolean mask and nonnegative coefficient")
    if not movable.any():
        return np.zeros(angles.shape[:-1], dtype=np.float64)
    return coefficient * np.mean((angles[..., movable] - neutral[movable])**2, axis=-1)


def vector_pairs(finger_count: int) -> tuple[tuple[int, int], ...]:
    """Directed tip pairs, then palm->tips; indices into [palm, tips]."""
    if type(finger_count) is not int or finger_count not in (3, 4, 5):
        raise ValueError("finger_count must be 3, 4 or 5")
    tips = range(1, finger_count + 1)
    return tuple((i, j) for i in tips for j in tips if i != j) + tuple((0, i) for i in tips)


@dataclass
class RewardOutput:
    reward: torch.Tensor
    cost: torch.Tensor
    squared_vector_errors: torch.Tensor


class DexPilotReward(nn.Module):
    """All inputs/outputs preserve their leading batch/time dimensions.

    Human angles: [..., 20] rad; robot points: [..., N+1, 3] metres.
    Robot frames [..., N+1, 3, 3] map local axes to the palm body's axes.
    Precomputed comparison also needs human fingertip frames [..., 5, 3, 3].
    Reward = -0.5 * mean_i ||robot_vector_i - reference_vector_i||^2.
    A masked-out token has zero cost/reward, and must be excluded from averaging.
    """

    def __init__(self, group: TrainingGroupConfig):
        super().__init__()
        group.require_ready()
        self.group = group
        self.human_fk = UmeTrackFK(side="right")
        self.pairs = vector_pairs(group.finger_count)
        self.register_buffer("human_indices", torch.tensor(
            (WRIST_INDEX, *(HUMAN_FINGERS.index(finger) for finger in group.human_fingers))))
        self.register_buffer("origins", torch.tensor([origin for origin, _ in self.pairs]))
        self.register_buffer("tasks", torch.tensor([task for _, task in self.pairs]))
        self.register_buffer("human_reference_offset", torch.tensor(group.human_reference_offset_m,
                                                                     dtype=torch.float64))
        self.register_buffer("human_palm_frame", self.human_fk.palm_frame(
            reference_finger=group.palm_reference_finger))

    @staticmethod
    def _mask(leading: torch.Size, valid_mask: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        if valid_mask is None:
            return torch.ones(leading, dtype=torch.bool, device=device)
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool or valid_mask.shape != leading:
            raise ValueError("valid_mask must be boolean and exactly match batch/time dimensions")
        return valid_mask.to(device)

    def compare(self, robot_points: torch.Tensor, human_landmarks: torch.Tensor, *,
                robot_frames: torch.Tensor, human_fingertip_frames: torch.Tensor,
                human_palm_frame: torch.Tensor | None = None,
                valid_mask: torch.Tensor | None = None) -> RewardOutput:
        """Compare FK geometry; an explicit palm basis re-expresses human axes.

        When provided, human_palm_frame must follow the same rigid coordinate
        change as the supplied human landmarks and fingertip frames. The fixed
        wrist-to-palm offset is re-expressed by that change automatically.
        """
        count = self.group.finger_count
        if (not isinstance(robot_points, torch.Tensor) or robot_points.ndim < 2
                or robot_points.shape[-2:] != (count + 1, 3) or not robot_points.is_floating_point()):
            raise ValueError(f"robot_points must be a floating tensor [..., {count+1}, 3] in metres")
        if (not isinstance(human_landmarks, torch.Tensor) or not human_landmarks.is_floating_point()
                or human_landmarks.shape != (*robot_points.shape[:-2], 21, 3)):
            raise ValueError("Human landmarks must match robot batch/time dimensions and have shape [..., 21, 3]")
        mask = self._mask(robot_points.shape[:-2], valid_mask, robot_points.device)
        leading = robot_points.shape[:-2]
        if (not isinstance(human_fingertip_frames, torch.Tensor)
                or human_fingertip_frames.shape != (*leading, 5, 3, 3)
                or not human_fingertip_frames.is_floating_point()):
            raise ValueError("Human fingertip frames must be floating [..., 5, 3, 3]")
        robot_frames = self._checked_frames(robot_frames, robot_points, count + 1, mask, "Robot")
        human_frames = human_fingertip_frames.to(robot_points)[..., self.human_indices[1:].to(robot_points.device), :, :]
        human_frames = self._checked_frames(human_frames, robot_points, count, mask, "Human")
        human = human_landmarks.to(robot_points)[..., self.human_indices.to(robot_points.device), :]
        robot = torch.where(mask[..., None, None], robot_points, 0)
        human = torch.where(mask[..., None, None], human, 0)
        if not torch.isfinite(robot).all() or not torch.isfinite(human).all():
            raise ValueError("Active reward points must be finite; invalid data must be masked out")
        # Both sides project directly into homologous local bases. The palm
        # basis is frozen at rest; distal bases follow their actual links.
        palm_frame = self.human_palm_frame if human_palm_frame is None else human_palm_frame
        if (not isinstance(palm_frame, torch.Tensor) or not palm_frame.is_floating_point()
                or palm_frame.shape not in ((3, 3), (*leading, 3, 3))):
            raise ValueError("Human palm frame must be floating [3,3] or match batch/time dimensions")
        palm_frame = palm_frame.to(robot_points).expand(*leading, 3, 3).unsqueeze(-3)
        palm_frame = self._checked_frames(palm_frame, robot_points, 1, mask, "Human palm")
        # Replace the wrist landmark by a fixed wrist-local palm reference.
        # UmeTrack's live palm landmark is skinned and would move with the fingers.
        reference_offset = self.human_reference_offset.to(robot_points)
        if human_palm_frame is not None:
            # An external change of human coordinate axes must also rotate the
            # fixed wrist-to-palm offset, not just the landmarks and tip frames.
            reexpression = palm_frame @ self.human_palm_frame.to(robot_points).T
            reference_offset = (reexpression @ reference_offset[:, None]).squeeze(-1)
        reference = human[..., :1, :] + reference_offset
        human = torch.cat((reference, human[..., 1:, :]), dim=-2)
        human = human - human[..., :1, :]
        human = human * self.group.human_scale
        # A nonzero fixed reference must not introduce error on padding tokens.
        human = torch.where(mask[..., None, None], human, 0)
        human_frames = torch.cat((palm_frame, human_frames), dim=-3)
        origins, tasks = self.origins.to(robot.device), self.tasks.to(robot.device)
        robot_vectors = robot[..., tasks, :] - robot[..., origins, :]
        human_vectors = human[..., tasks, :] - human[..., origins, :]
        # Row vectors right-multiply the local->palm basis to enter local axes.
        robot_vectors = (robot_vectors.unsqueeze(-2) @ robot_frames[..., origins, :, :]).squeeze(-2)
        human_vectors = (human_vectors.unsqueeze(-2) @ human_frames[..., origins, :, :]).squeeze(-2)
        squared = (robot_vectors - human_vectors).square().sum(dim=-1)
        cost = 0.5 * squared.mean(dim=-1)
        return RewardOutput(-cost, cost, squared)

    @staticmethod
    def _checked_frames(value: torch.Tensor, points: torch.Tensor, count: int,
                        mask: torch.Tensor, name: str) -> torch.Tensor:
        expected = (*points.shape[:-2], count, 3, 3)
        if (not isinstance(value, torch.Tensor) or value.shape != expected
                or not value.is_floating_point()):
            raise ValueError(f"{name} frames must be floating {expected}")
        identity = torch.eye(3, dtype=points.dtype, device=points.device).expand(expected)
        value = torch.where(mask[..., None, None, None], value.to(points), identity)
        check = value.detach().to(dtype=torch.float64)
        target = identity.to(dtype=torch.float64)
        if (not torch.isfinite(check).all()
                or not torch.allclose(check.transpose(-1, -2) @ check, target, atol=1e-5, rtol=0)
                or not torch.allclose(torch.linalg.det(check), torch.ones_like(check[..., 0, 0]),
                                      atol=1e-5, rtol=0)):
            raise ValueError(f"{name} frames must be proper rotations, not reflections or degenerate axes")
        return value

    def forward(self, human_angles: torch.Tensor, robot_points: torch.Tensor, *,
                robot_frames: torch.Tensor,
                valid_mask: torch.Tensor | None = None) -> RewardOutput:
        if not isinstance(human_angles, torch.Tensor) or human_angles.ndim < 1 or human_angles.shape[-1] != 20:
            raise ValueError("human_angles must have shape [..., 20]")
        mask = self._mask(human_angles.shape[:-1], valid_mask, human_angles.device)
        safe_angles = torch.where(mask[..., None], human_angles, 0)
        return self.compare(robot_points, self.human_fk(safe_angles), robot_frames=robot_frames,
                            human_fingertip_frames=self.human_fk.fingertip_frames(safe_angles),
                            valid_mask=valid_mask)


@dataclass(frozen=True)
class StateReward:
    reward: float
    cost: float
    squared_vector_errors: np.ndarray
    simulation_time: float
    query_reference_time: float
    sampled_reference_time: float
    reference_index: int


class HandMotionReward:
    """Bind one model's landmark definition and one continuous reference window.

    reference_start_time explicitly maps simulation time 0 (after reset) to a
    recording timestamp. A step's reward uses its END time, including servo lag.
    Values are environment rewards, detached from the policy-gradient graph.
    """

    def __init__(self, group: TrainingGroupConfig, landmarks: RobotLandmarks,
                 reference: HumanReference, *, reference_start_time: float,
                 max_time_error_seconds: float):
        if landmarks.finger_ids != group.robot_finger_ids:
            raise ValueError("Robot landmark order differs from the training group's finger correspondence")
        landmarks.require_finger_frames()
        if not np.isfinite(reference_start_time) or not 0 <= max_time_error_seconds < float("inf"):
            raise ValueError("Provide a finite reference_start_time and nonnegative finite time tolerance")
        # Catch an invalid episode origin before any simulation advances.
        reference.sample(reference_start_time, max_error_seconds=max_time_error_seconds)
        self.group, self.landmarks, self.reference = group, landmarks, reference
        self.reference_start_time = float(reference_start_time)
        self.max_time_error_seconds = float(max_time_error_seconds)
        self.metric = DexPilotReward(group)
        # Calibrate before stepping, so missing geometry or degenerate axes
        # cannot silently produce a training reward in the wrong palm frame.
        landmarks.palm_rest_frame(reference_finger=group.robot_palm_reference_finger)
        # Reference geometry does not depend on a sampled action or candidate hand.
        with torch.no_grad():
            self.reference_landmarks = self.metric.human_fk(reference.angles)
            self.reference_frames = self.metric.human_fk.fingertip_frames(reference.angles)

    def __call__(self, state: HandState) -> StateReward:
        if not np.isfinite(state.time) or state.time < 0:
            raise ValueError("State must have a finite nonnegative simulation time")
        query = self.reference_start_time + state.time
        selected = self.reference.sample(query, max_error_seconds=self.max_time_error_seconds)
        index = int(selected.indices)
        human = self.reference_landmarks[index]
        robot = torch.as_tensor(self.landmarks.positions(state), dtype=human.dtype, device=human.device)
        robot_frames = torch.as_tensor(self.landmarks.orientations(
            state, palm_finger=self.group.robot_palm_reference_finger), dtype=human.dtype, device=human.device)
        with torch.no_grad():
            output = self.metric.compare(robot, human, robot_frames=robot_frames,
                                         human_fingertip_frames=self.reference_frames[index])
        return StateReward(float(output.reward), float(output.cost),
                           output.squared_vector_errors.cpu().numpy().copy(), float(state.time),
                           query, float(selected.time), index)
