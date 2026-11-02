"""Compiled HOD hand -> canonical 5 fingers x 4 joint/actuator slots.

The compiled model, not optional HOD metadata, defines limits and actuation.
Slots follow HOD finger indices, not anatomical thumb/index/etc. correspondence.
The policy descriptor follows HOD's 35-D presence/grammar representation;
joint limits remain separate execution constraints.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np


SLOT_COUNT = 20
JOINT_ROLES = ("palm", "base", "middle", "end")
GHOST_RANGE_TOLERANCE = 1e-5  # Same tolerance as the existing HOD validation wrapper.
HOD_MORPHOLOGY_DIM = 35


def hod_policy_features(finger_tokens: Sequence[str]) -> np.ndarray:
    """Encode five HOD filename tokens with the upstream feature ordering.

    See HOD revision 4b86c809828a1c24c30899abc75a293bf3bca56b,
    IsaacLab/source/codesign/codesign/utils/morph_vector.py:
    parse_morph_from_filename() and build_hand_morph_obs().

    An empty token means an absent finger; "0" is one real distal joint.
    Digits are right-aligned to base/middle/end, with scale (digit + 1) / 10.
    Layout: five finger-presence flags, then six values per finger:
    three joint-presence flags followed by three normalized grammar scales.
    These scales are grammar codes, not physical lengths in metres.
    """
    if isinstance(finger_tokens, str) or len(finger_tokens) != 5:
        raise ValueError("finger_tokens must contain exactly five strings in f1..f5 order")
    scales = np.zeros((5, 3), dtype=np.float32)
    for finger, token in enumerate(finger_tokens):
        if not isinstance(token, str) or re.fullmatch(r"[0-9]{0,3}", token) is None:
            raise ValueError("Each finger token must be empty or contain one to three ASCII digits")
        if token:
            scales[finger, -len(token):] = [(int(digit) + 1) / 10 for digit in token]
    joint_presence = scales > 0
    finger_presence = joint_presence.any(axis=1).astype(np.float32)
    per_finger = np.concatenate((joint_presence, scales), axis=1)
    return np.concatenate((finger_presence, per_finger.ravel())).astype(np.float32)


@dataclass
class HandMorphology:
    model: mujoco.MjModel
    source: Path | None
    joint_names: tuple[str | None, ...]
    joint_ids: np.ndarray
    actuator_ids: np.ndarray
    qpos_ids: np.ndarray
    qvel_ids: np.ndarray
    active_mask: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def load(cls, mjcf: str | Path, *, slot_joint_names: Sequence[str | None] | None = None):
        source = Path(mjcf).resolve()
        model = mujoco.MjModel.from_xml_path(str(source))
        return cls.from_model(model, source=source, slot_joint_names=slot_joint_names)

    @classmethod
    def from_model(cls, model: mujoco.MjModel, *, source: Path | None = None,
                   slot_joint_names: Sequence[str | None] | None = None):
        if slot_joint_names is None:
            slot_joint_names = [None] * SLOT_COUNT
            for joint_id in range(model.njnt):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                match = re.fullmatch(r"f([1-5])_finger_(palm|base|middle|end)_joint", name or "")
                if match:
                    slot = (int(match[1]) - 1) * 4 + JOINT_ROLES.index(match[2])
                    slot_joint_names[slot] = name
        if len(slot_joint_names) != SLOT_COUNT:
            raise ValueError("slot_joint_names must contain exactly 20 names/None entries")
        present = [name for name in slot_joint_names if name is not None]
        if not present or len(set(present)) != len(present):
            raise ValueError("Need unique hand joint names; supply an explicit mapping for non-HOD names")
        joint_ids, actuator_ids, qpos_ids, qvel_ids = [np.full(SLOT_COUNT, -1, dtype=np.int64)
                                                     for _ in range(4)]
        mask = np.zeros(SLOT_COUNT, dtype=bool)
        lower, upper = np.zeros(SLOT_COUNT), np.zeros(SLOT_COUNT)
        used_actuators = set()
        for slot, name in enumerate(slot_joint_names):
            if name is None:
                continue
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Mapped joint does not exist: {name}")
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_limited[joint_id]:
                raise ValueError(f"Hand joint must be a limited hinge: {name}")
            matches = [a for a in range(model.nu)
                       if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                       and model.actuator_trnid[a, 0] == joint_id]
            joint_lo, joint_hi = model.jnt_range[joint_id]
            if not matches and joint_hi - joint_lo <= GHOST_RANGE_TOLERANCE and joint_lo <= 0 <= joint_hi:
                # Some exports omit servos for fixed ghost placeholders.
                joint_ids[slot] = joint_id
                qpos_ids[slot], qvel_ids[slot] = model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id]
                continue
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one direct position actuator for {name}")
            actuator = matches[0]
            gain, bias = model.actuator_gainprm[actuator], model.actuator_biasprm[actuator]
            if (model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
                    or model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE
                    or model.actuator_dyntype[actuator] != mujoco.mjtDyn.mjDYN_NONE
                    or gain[0] <= 0 or not np.isclose(bias[0], 0)
                    or not np.isclose(bias[1], -gain[0]) or bias[2] > 0
                    or not np.allclose(model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0])):
                raise ValueError(f"Actuator for {name} is not a supported unit-gear position servo")
            lo, hi = model.jnt_range[joint_id]
            if model.actuator_ctrllimited[actuator]:
                lo = max(lo, model.actuator_ctrlrange[actuator, 0])
                hi = min(hi, model.actuator_ctrlrange[actuator, 1])
            if not np.isfinite([lo, hi]).all() or hi < lo:
                raise ValueError(f"Invalid or incompatible joint/control limits for {name}")
            joint_ids[slot], actuator_ids[slot] = joint_id, actuator
            qpos_ids[slot], qvel_ids[slot] = model.jnt_qposadr[joint_id], model.jnt_dofadr[joint_id]
            used_actuators.add(actuator)
            if hi - lo > GHOST_RANGE_TOLERANCE:
                mask[slot], lower[slot], upper[slot] = True, lo, hi
            elif not lo <= 0 <= hi:
                raise ValueError(f"Ghost joint must admit its zero placeholder: {name}")
        if used_actuators != set(range(model.nu)):
            raise ValueError("Model has actuators outside the supplied 20-slot hand mapping")
        if not mask.any():
            raise ValueError("Hand has no active position actuators")
        return cls(model, source, tuple(slot_joint_names), joint_ids, actuator_ids,
                   qpos_ids, qvel_ids, mask, lower, upper)

    @property
    def active_count(self) -> int:
        return int(self.active_mask.sum())

    def policy_features(self, *, finger_tokens: Sequence[str] | None = None) -> np.ndarray:
        """Return the HOD 35-D policy descriptor, checking the compiled mask.

        Standard HOD filenames carry all five grammar tokens. For renamed or
        in-memory models, the caller supplies the original tokens explicitly;
        geometry is never guessed from joint limits. Palm layout and fingertip
        type are not encoded by this descriptor.
        """
        if finger_tokens is None:
            matches = [] if self.source is None else re.findall(
                r"(?:^|_)f([1-5])_([0-9]*)(?=_|$)", self.source.stem)
            if len(matches) != 5 or {finger for finger, _ in matches} != set("12345"):
                raise ValueError("HOD grammar tokens unavailable in filename; pass finger_tokens explicitly")
            tokens = dict(matches)
            finger_tokens = [tokens[str(finger)] for finger in range(1, 6)]
        features = hod_policy_features(finger_tokens)
        per_finger = features[5:].reshape(5, 6)
        expected_mask = np.column_stack((features[:5], per_finger[:, :3])).ravel().astype(bool)
        if not np.array_equal(expected_mask, self.active_mask):
            raise ValueError("HOD grammar tokens disagree with the compiled active joint slots")
        return features

    def validate_angles(self, values: np.ndarray, *, name: str = "joint_angles") -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (SLOT_COUNT,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a finite 20-slot vector in radians")
        if np.any(values < self.lower) or np.any(values > self.upper):
            raise ValueError(f"{name} violates limits or assigns a nonzero ghost slot")
        return values.copy()
