"""Shared hand correspondence and reward definition for one controller group."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

HUMAN_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class TrainingGroupConfig:
    name: str
    robot_finger_ids: tuple[int, ...]
    human_fingers: tuple[str, ...]
    human_scale: float | None
    # Fixed point in the right-hand wrist frame, in metres. None is unconfigured.
    human_reference_offset_m: tuple[float, float, float] | None = None
    palm_reference_finger: str = "middle"
    reward_definition: str = "origin_local_bidirectional_palm_v2"

    def __post_init__(self):
        if self.reward_definition != "origin_local_bidirectional_palm_v2":
            raise ValueError("Unsupported reward definition; expected bidirectional vectors with semantic palm frames")
        object.__setattr__(self, "robot_finger_ids", tuple(self.robot_finger_ids))
        object.__setattr__(self, "human_fingers", tuple(self.human_fingers))
        count = len(self.robot_finger_ids)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Training group needs a nonempty name")
        if (count not in (3, 4, 5) or len(self.human_fingers) != count
                or len(set(self.robot_finger_ids)) != count
                or any(type(i) is not int or not 1 <= i <= 5 for i in self.robot_finger_ids)
                or len(set(self.human_fingers)) != count
                or any(f not in HUMAN_FINGERS for f in self.human_fingers)
                or self.human_fingers[0] != "thumb"):
            raise ValueError("Need 3-5 unique robot fingers (1..5) and human fingers, thumb first")
        if self.name in ("sym3", "sym4", "sym5") and count != int(self.name[-1]):
            raise ValueError("Symmetric group name disagrees with finger count")
        if self.human_scale is not None:
            if (isinstance(self.human_scale, bool) or not np.isscalar(self.human_scale)
                    or not np.isfinite(self.human_scale) or self.human_scale <= 0):
                raise ValueError("human_scale must be a finite positive scalar")
            object.__setattr__(self, "human_scale", float(self.human_scale))
        if not isinstance(self.palm_reference_finger, str) or self.palm_reference_finger not in self.human_fingers:
            raise ValueError("palm_reference_finger must be included in the group's human fingers")
        if self.human_reference_offset_m is not None:
            offset = np.asarray(self.human_reference_offset_m, dtype=float)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError("human_reference_offset_m must be a finite wrist-local point [3] in metres")
            object.__setattr__(self, "human_reference_offset_m", tuple(offset.tolist()))

    @property
    def finger_count(self) -> int:
        return len(self.robot_finger_ids)

    @property
    def robot_palm_reference_finger(self) -> int:
        return self.robot_finger_ids[self.human_fingers.index(self.palm_reference_finger)]

    def require_ready(self) -> None:
        missing = [name for name in ("human_scale", "human_reference_offset_m")
                   if getattr(self, name) is None]
        if missing:
            raise ValueError(f"Set {', '.join(missing)} explicitly before computing reward")

    def to_dict(self) -> dict:
        # JSON-compatible lists also make checkpoint metadata independent of tuple/list spelling.
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value: dict) -> TrainingGroupConfig:
        if "human_to_robot_rotation" in value:
            raise ValueError("Legacy global-rotation groups must be replaced by semantic palm-frame configuration")
        return cls(**value)


def load_training_groups(path: str | Path) -> dict[str, TrainingGroupConfig]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(value) != {"schema_version", "groups"} or value["schema_version"] != 2:
        raise ValueError("Unsupported training-group configuration schema")
    configs = [TrainingGroupConfig.from_dict(group) for group in value["groups"]]
    result = {group.name: group for group in configs}
    if not result or len(result) != len(configs):
        raise ValueError("Training groups must have unique names")
    return result
