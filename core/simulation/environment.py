"""Incremental position control for any hand accepted by HandMorphology.

No hand path, control frequency, initial pose, reward or training algorithm is
fixed here. MuJoCo state is NumPy and is not a differentiable PyTorch graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from simulation.morphology import HandMorphology, SLOT_COUNT


@dataclass
class HandState:
    time: float
    joint_angles: np.ndarray       # [20], radians; inactive slots zero
    joint_velocities: np.ndarray   # [20], radians/s
    target_angles: np.ndarray      # [20], commanded positions, not actual motion
    active_mask: np.ndarray        # [20]
    joint_anchors_world: np.ndarray  # [20, 3], hinge anchors, not fingertips
    body_names: tuple[str | None, ...]
    body_positions_world: np.ndarray
    body_quaternions_world: np.ndarray
    site_names: tuple[str | None, ...]
    site_positions_world: np.ndarray


@dataclass
class HandStep:
    state: HandState
    requested_increment: np.ndarray
    applied_target_increment: np.ndarray
    clipped_slots: np.ndarray
    physics_steps: int


class SimulationError(RuntimeError):
    pass


class HandEnvironment:
    def __init__(self, hand: HandMorphology | str | Path, *,
                 max_joint_increment: float | np.ndarray | None = None):
        self.hand = hand if isinstance(hand, HandMorphology) else HandMorphology.load(hand)
        self.model = self.hand.model
        self.data = mujoco.MjData(self.model)
        if max_joint_increment is None:
            self.max_joint_increment = None
        else:
            limit = np.asarray(max_joint_increment, dtype=np.float64)
            if limit.ndim == 0:
                limit = np.full(SLOT_COUNT, float(limit))
            if limit.shape != (SLOT_COUNT,) or not np.isfinite(limit).all() or np.any(limit <= 0):
                raise ValueError("max_joint_increment must be positive radians, scalar or 20-slot vector")
            self.max_joint_increment = limit.copy()
        self.target = np.zeros(SLOT_COUNT)
        self._ready = False
        self._body_names = tuple(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                                 for i in range(self.model.nbody))
        self._site_names = tuple(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
                                 for i in range(self.model.nsite))

    @property
    def physics_timestep(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self, joint_angles: np.ndarray, *, target_angles: np.ndarray | None = None) -> HandState:
        """Reset with an explicitly supplied initial pose; velocities become zero.

        Other scene state follows the loaded model's reset defaults. Target
        defaults to the supplied pose, rather than accumulating an old target.
        """
        joint_angles = self.hand.validate_angles(joint_angles)
        target = joint_angles if target_angles is None else self.hand.validate_angles(
            target_angles, name="target_angles")
        self._ready = False
        mujoco.mj_resetData(self.model, self.data)
        active = self.hand.active_mask
        self.data.qpos[self.hand.qpos_ids[active]] = joint_angles[active]
        self.target = target.copy()
        self._write_controls()
        mujoco.mj_forward(self.model, self.data)
        self._assert_healthy()
        self._ready = True
        return self.state()

    def _write_controls(self) -> None:
        present = self.hand.actuator_ids >= 0
        self.data.ctrl[self.hand.actuator_ids[present]] = self.target[present]

    def _assert_healthy(self, expected_time: float | None = None) -> None:
        finite = all(np.isfinite(value).all() for value in (self.data.qpos, self.data.qvel,
                                                           self.data.act, self.data.ctrl))
        bad_time = expected_time is not None and not np.isclose(
            self.data.time, expected_time, rtol=1e-9, atol=1e-10)
        if not finite or any(warning.number for warning in self.data.warning) or bad_time:
            self._ready = False
            raise SimulationError("MuJoCo reported a warning, nonfinite state, or unexpected time reset")

    def state(self) -> HandState:
        if not self._ready:
            raise RuntimeError("Call reset with an explicit initial pose before reading or stepping")
        active = self.hand.active_mask
        angles, velocities = np.zeros(SLOT_COUNT), np.zeros(SLOT_COUNT)
        anchors = np.zeros((SLOT_COUNT, 3))
        angles[active] = self.data.qpos[self.hand.qpos_ids[active]]
        velocities[active] = self.data.qvel[self.hand.qvel_ids[active]]
        anchors[active] = self.data.xanchor[self.hand.joint_ids[active]]
        # Copies are necessary: mj_step modifies the backing arrays in place.
        return HandState(float(self.data.time), angles, velocities, self.target.copy(), active.copy(),
                         anchors, self._body_names, self.data.xpos.copy(), self.data.xquat.copy(),
                         self._site_names, self.data.site_xpos.copy())

    def step(self, increment: np.ndarray, *, physics_steps: int) -> HandStep:
        """Accumulate one 20-slot increment, then hold it for physics_steps.

        The caller chooses control timing. One call advances exactly
        physics_steps * physics_timestep seconds. Limits act on commanded
        targets; soft physical joint constraints may allow actual overshoot.
        """
        if not self._ready:
            raise RuntimeError("Call reset with an explicit initial pose before stepping")
        increment = np.asarray(increment, dtype=np.float64)
        if increment.shape != (SLOT_COUNT,) or not np.isfinite(increment).all():
            raise ValueError("increment must be a finite 20-slot vector in radians")
        if type(physics_steps) is not int or physics_steps <= 0:
            raise ValueError("physics_steps must be a positive integer")
        requested = increment.copy()
        delta = np.where(self.hand.active_mask, increment, 0)
        if self.max_joint_increment is not None:
            delta = np.clip(delta, -self.max_joint_increment, self.max_joint_increment)
        previous = self.target.copy()
        self.target = np.clip(previous + delta, self.hand.lower, self.hand.upper)
        self._write_controls()
        before_time = float(self.data.time)
        mujoco.mj_step(self.model, self.data, nstep=physics_steps)
        mujoco.mj_forward(self.model, self.data)
        expected_time = before_time + physics_steps * self.physics_timestep
        self._assert_healthy(expected_time)
        applied = self.target - previous
        clipped = ~np.isclose(applied, requested, rtol=0, atol=1e-12)
        return HandStep(self.state(), requested, applied, clipped, physics_steps)
