"""Explicit sites/body offsets -> actual robot landmarks in the palm frame.

No HOD body origin is implicitly treated as a fingertip. Bindings are supplied
for the loaded model and use the same finger IDs as the group's correspondence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np

from simulation.environment import HandState
from simulation.morphology import HandMorphology
from .frames import finger_frame


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


@dataclass(frozen=True)
class PointSpec:
    body: str | None = None
    offset_m: tuple[float, float, float] | None = None
    site: str | None = None

    def __post_init__(self):
        if (self.body is None) == (self.site is None):
            raise ValueError("A landmark must specify either a body with offset_m, or a site")
        if self.body is not None:
            offset = np.asarray(self.offset_m, dtype=float)
            if not self.body or offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError("Body landmarks require an explicit finite offset_m [3]")
            object.__setattr__(self, "offset_m", tuple(offset.tolist()))
        elif not self.site or self.offset_m is not None:
            raise ValueError("A site landmark uses its site position and must not specify offset_m")

    @classmethod
    def load(cls, path: str | Path) -> PointSpec:
        """Load one body-local point or site binding, e.g. a saved palm reference."""
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class FingerFrameSpec:
    """Two fixed axes in a distal rigid body; origin is the separate tip point."""
    body: str
    longitudinal_axis: tuple[float, float, float]
    flexion_axis: tuple[float, float, float]

    def __post_init__(self):
        if not isinstance(self.body, str) or not self.body:
            raise ValueError("Finger frame requires a named distal body")
        basis = finger_frame(self.longitudinal_axis, self.flexion_axis)
        object.__setattr__(self, "longitudinal_axis", tuple(basis[:, 0].tolist()))
        object.__setattr__(self, "flexion_axis", tuple(basis[:, 1].tolist()))

    @property
    def rotation(self) -> np.ndarray:
        x, y = np.array(self.longitudinal_axis), np.array(self.flexion_axis)
        return np.column_stack((x, y, np.cross(x, y)))

    @classmethod
    def from_joint(cls, hand: HandMorphology, tip: PointSpec, *, joint: str) -> FingerFrameSpec:
        """Calibrate once from an explicit tip and its distal flexion hinge.

        Missing/fixed-away joints require explicitly supplied body axes instead
        of guessing another joint. A surviving zero-range hinge is sufficient.
        """
        model = hand.model
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("Finger frame calibration needs an existing named flexion hinge")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        if tip.body is not None:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tip.body)
            if body_id < 0:
                raise ValueError("Finger frame tip body is missing")
            point = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ np.array(tip.offset_m)
        else:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip.site)
            if site_id < 0:
                raise ValueError("Finger frame tip site is missing")
            body_id = int(model.site_bodyid[site_id])
            point = data.site_xpos[site_id]
        joint_body = int(model.jnt_bodyid[joint_id])
        current = body_id
        while current != joint_body:
            if current == 0 or model.body_jntnum[current]:
                raise ValueError("Tip must be rigidly attached on the distal side of the selected hinge")
            current = int(model.body_parentid[current])
        if model.body_jntnum[joint_body] != 1:
            raise ValueError("A body with multiple joints needs explicit finger frame axes")
        world_from_body = data.xmat[body_id].reshape(3, 3)
        direction = world_from_body.T @ (point - data.xanchor[joint_id])
        axis = world_from_body.T @ data.xaxis[joint_id]
        return cls(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                   tuple(direction.tolist()), tuple(axis.tolist()))


class RobotLandmarks:
    def __init__(self, hand: HandMorphology, *, palm_frame: str, palm: PointSpec,
                 fingertips: Mapping[int, PointSpec],
                 fingertip_frames: Mapping[int, FingerFrameSpec] | None = None):
        self.hand = hand
        self.finger_ids = tuple(fingertips)
        active = tuple((np.flatnonzero(hand.active_mask.reshape(5, 4).any(axis=1)) + 1).tolist())
        if (not 3 <= len(self.finger_ids) <= 5 or set(self.finger_ids) != set(active)
                or any(type(i) is not int for i in self.finger_ids)):
            raise ValueError("Landmark finger IDs must exactly match the model's active fingers")
        self.body_names = tuple(mujoco.mj_id2name(hand.model, mujoco.mjtObj.mjOBJ_BODY, i)
                                for i in range(hand.model.nbody))
        self.site_names = tuple(mujoco.mj_id2name(hand.model, mujoco.mjtObj.mjOBJ_SITE, i)
                                for i in range(hand.model.nsite))
        if palm_frame not in self.body_names:
            raise ValueError(f"Missing palm coordinate frame body: {palm_frame}")
        self.frame_id = self.body_names.index(palm_frame)
        self.points = tuple(self._bind(spec) for spec in (palm, *fingertips.values()))
        self._palm_rest_frames = {}
        self.finger_frames = None
        if fingertip_frames is not None:
            if set(fingertip_frames) != set(self.finger_ids):
                raise ValueError("Finger frame IDs must match the fingertip bindings")
            bindings = []
            for finger in self.finger_ids:
                spec = fingertip_frames[finger]
                if spec.body not in self.body_names:
                    raise ValueError(f"Missing finger frame body: {spec.body}")
                bindings.append((self.body_names.index(spec.body), spec.rotation))
            self.finger_frames = tuple(bindings)

    def _bind(self, spec: PointSpec) -> tuple[str, int, np.ndarray | None]:
        if spec.body is not None:
            if spec.body not in self.body_names:
                raise ValueError(f"Missing landmark body: {spec.body}")
            return "body", self.body_names.index(spec.body), np.array(spec.offset_m)
        if spec.site not in self.site_names:
            raise ValueError(f"Missing landmark site: {spec.site}")
        return "site", self.site_names.index(spec.site), None

    @classmethod
    def from_dict(cls, hand: HandMorphology, value: dict) -> RobotLandmarks:
        required = {"palm_frame", "palm", "fingertips"}
        if not required <= set(value) or set(value) - required - {"fingertip_frames"}:
            raise ValueError("Landmark configuration needs palm_frame, palm, fingertips, optional fingertip_frames")
        tips = {int(i): PointSpec(**spec) for i, spec in value["fingertips"].items()}
        frames = value.get("fingertip_frames")
        if frames is not None:
            if {int(i) for i in frames} != set(tips):
                raise ValueError("Finger frame IDs must match the fingertip bindings")
            frames = {int(i): (FingerFrameSpec.from_joint(hand, tips[int(i)], joint=spec["joint"])
                               if set(spec) == {"joint"} else FingerFrameSpec(**spec))
                      for i, spec in frames.items()}
        return cls(hand, palm_frame=value["palm_frame"], palm=PointSpec(**value["palm"]),
                   fingertips=tips, fingertip_frames=frames)

    @classmethod
    def load(cls, hand: HandMorphology, path: str | Path) -> RobotLandmarks:
        return cls.from_dict(hand, json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_hod(cls, hand: HandMorphology) -> RobotLandmarks:
        """Bind the loaded HOD candidate's own palm and fingertip geometry."""
        from .hod import hod_landmark_config
        return cls.from_dict(hand, hod_landmark_config(hand))

    def positions(self, state: HandState) -> np.ndarray:
        """Return [palm, tips in finger_ids order], [N+1, 3], in metres.

        Uses the snapshot's actual body/site poses, including rotation of body
        offsets. Commanded target angles are never used to compute these points.
        """
        self._check_state(state)
        points = []
        for kind, index, offset in self.points:
            if kind == "site":
                points.append(state.site_positions_world[index])
            else:
                points.append(state.body_positions_world[index]
                              + _rotation(state.body_quaternions_world[index]) @ offset)
        points = np.stack(points)
        frame = _rotation(state.body_quaternions_world[self.frame_id])
        local = (points - points[0]) @ frame
        if not np.isfinite(local).all():
            raise ValueError("Nonfinite actual robot landmark positions")
        return local

    def _check_state(self, state: HandState) -> None:
        if (state.body_names != self.body_names or state.site_names != self.site_names
                or not np.array_equal(state.active_mask, self.hand.active_mask)):
            raise ValueError("State does not match this robot landmark binding")

    def require_finger_frames(self) -> None:
        if self.finger_frames is None:
            raise ValueError("Local-vector reward needs explicit fingertip_frames for this hand")

    def palm_rest_frame(self, *, reference_finger: int) -> np.ndarray:
        """Fixed semantic palm-local -> native palm-body basis at zero angles.

        Copy the corresponding finger's distal basis at rest, express it in the
        palm body, and freeze it there. Finger actions cannot move this basis;
        neither the palm point's depth nor CAD axis labels define its direction.
        """
        self.require_finger_frames()
        if type(reference_finger) is not int or reference_finger not in self.finger_ids:
            raise ValueError("Palm reference finger must be one of the bound fingers")
        if reference_finger not in self._palm_rest_frames:
            model = self.hand.model
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            distal_body, distal_frame = self.finger_frames[self.finger_ids.index(reference_finger)]
            native_from_world = data.xmat[self.frame_id].reshape(3, 3).T
            self._palm_rest_frames[reference_finger] = (
                native_from_world @ data.xmat[distal_body].reshape(3, 3) @ distal_frame)
        return self._palm_rest_frames[reference_finger].copy()

    def orientations(self, state: HandState, *, palm_finger: int) -> np.ndarray:
        """Return [palm, tips] proper rotation matrices in the palm body's axes.

        Frame origins are the positions() points. Basis vectors are transported
        by actual body poses, including fixed links or absent actuated joints.
        The palm's neutral basis stays fixed in its body; palm_finger comes
        from the group's human-to-robot correspondence, not a guessed slot.
        """
        self._check_state(state)
        self.require_finger_frames()
        world_from_palm = _rotation(state.body_quaternions_world[self.frame_id])
        frames = [self.palm_rest_frame(reference_finger=palm_finger)]
        for body_id, local_frame in self.finger_frames:
            frames.append(world_from_palm.T @ _rotation(state.body_quaternions_world[body_id]) @ local_frame)
        result = np.stack(frames)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite actual finger orientations")
        return result
