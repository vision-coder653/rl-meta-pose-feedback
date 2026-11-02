"""Pose-only DexPilot-style retargeting with scaled targets and an open-pose prior.

The geometric metric is shared with training; scale and the angle prior below
belong to the retarget optimizer. No pinch projection or dynamics is introduced.
Each call is independent. A failed result is diagnostic, never a reset pose.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time

import numpy as np
from scipy.optimize import minimize
import torch

from simulation.environment import HandEnvironment
from configs.groups import TrainingGroupConfig
from kinematics import RobotLandmarks
from simulation.morphology import HandMorphology
from simulation.reward import DexPilotReward, RewardOutput, open_pose_cost


@dataclass(frozen=True)
class RetargetingConfig:
    human_scale: float | None = 2.4  # Absolute scale; None inherits the training group.
    open_regularization: float = .0025  # lambda * mean(active angle offset^2), radians.
    seed: int = 42
    random_starts: int = 2
    max_iterations: int = 200
    ftol: float = 1e-9
    collision_ftol: float = 1e-2  # Mixed cm^2 objective / mm constraint stopping tolerance.
    difference_step: float = 1e-6
    objective_scale: float = 10000.0  # m^2 -> cm^2; minimizer is unchanged.
    penetration_tolerance_m: float = .012
    reject_colliding_best: bool = True  # Preserve the best match; do not substitute a worse feasible pose.
    collision_refinement: bool = True  # Recovery only when reject_colliding_best=False.

    def resolve_group(self, group: TrainingGroupConfig) -> TrainingGroupConfig:
        """Use the same absolute human scale in initialization and PPO reward."""
        return group if self.human_scale is None else replace(group, human_scale=self.human_scale)

    def __post_init__(self):
        if type(self.reject_colliding_best) is not bool:
            raise ValueError("reject_colliding_best must be boolean")
        if self.human_scale is not None and (not np.isfinite(self.human_scale) or self.human_scale <= 0):
            raise ValueError("human_scale must be finite and positive, or None to inherit the group")
        if not np.isfinite(self.open_regularization) or self.open_regularization < 0:
            raise ValueError("open_regularization must be finite and nonnegative")
        if type(self.collision_refinement) is not bool:
            raise ValueError("collision_refinement must be boolean")
        for name in ("seed", "random_starts", "max_iterations"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "max_iterations" else 0):
                raise ValueError(f"Invalid {name}")
        for name in ("ftol", "collision_ftol", "difference_step", "objective_scale", "penetration_tolerance_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class RetargetingResult:
    accepted: bool
    status: str
    candidate_angles: np.ndarray
    initial_cost: float
    cost: float
    squared_vector_errors: np.ndarray
    attempts: list[dict]
    selected_attempt: int
    contacts: list[dict]
    elapsed_seconds: float
    geometric_cost: float = 0.
    regularization_cost: float = 0.
    initial_geometric_cost: float = 0.
    initial_regularization_cost: float = 0.
    human_scale: float = 1.

    def require_angles(self) -> np.ndarray:
        if not self.accepted:
            raise RuntimeError(f"Retargeting did not produce an accepted pose: {self.status}")
        return self.candidate_angles.copy()

    def to_dict(self) -> dict:
        result = asdict(self)
        result["candidate_angles"] = self.candidate_angles.tolist()
        result["squared_vector_errors"] = self.squared_vector_errors.tolist()
        result["local_vector_rms_mm"] = float(np.sqrt(self.squared_vector_errors.mean()) * 1000)
        return result


class PoseRetargeter:
    """Not thread-safe; owns an independent MjData, never changes the model.

    Contact acceptance only covers collision pairs enabled in the MJCF. It is
    not a visual-mesh intersection guarantee or a dynamic stability certificate.
    """

    def __init__(self, hand: HandMorphology, landmarks: RobotLandmarks,
                 group: TrainingGroupConfig, config: RetargetingConfig | None = None):
        if landmarks.hand is not hand:
            raise ValueError("Landmarks must belong to the supplied hand")
        if landmarks.finger_ids != group.robot_finger_ids:
            raise ValueError("Hand fingers differ from the group's correspondence")
        self.config = config or RetargetingConfig()
        group = self.config.resolve_group(group)
        self.hand, self.landmarks, self.group = hand, landmarks, group
        self.metric = DexPilotReward(group).double().eval()
        self.environment = HandEnvironment(hand)
        landmarks.palm_rest_frame(reference_finger=group.robot_palm_reference_finger)
        self.movable = hand.active_mask & (hand.upper > hand.lower)
        self.neutral = np.clip(np.zeros(20), hand.lower, hand.upper)

    def regularization_cost(self, angles) -> float:
        """Favor the legal zero/open pose, averaging only over movable joints."""
        return float(open_pose_cost(angles, self.neutral, self.movable, self.config.open_regularization))

    @torch.no_grad()
    def human_geometry(self, angles):
        angles = torch.as_tensor(angles, dtype=torch.float64).detach().cpu()
        if angles.shape != (20,) or not torch.isfinite(angles).all():
            raise ValueError("Human pose must be a finite 20-angle vector in radians")
        return self.metric.human_fk(angles), self.metric.human_fk.fingertip_frames(angles)

    def geometry(self, angles):
        state = self.environment.reset(np.asarray(angles, dtype=np.float64))
        return (self.landmarks.positions(state), self.landmarks.orientations(
            state, palm_finger=self.group.robot_palm_reference_finger))

    def contacts(self) -> list[dict]:
        """Read enabled collision contacts after the most recent FK reset."""
        model, data = self.hand.model, self.environment.data
        result = []
        for contact in data.contact:
            if contact.dist >= 0:
                continue
            first, second = int(contact.geom1), int(contact.geom2)
            result.append({"geom1": model.geom(first).name, "geom2": model.geom(second).name,
                           "penetration_m": float(-contact.dist)})
        return result

    @torch.no_grad()
    def solve(self, human_angles, *, initial_angles=None) -> RetargetingResult:
        points, frames = self.human_geometry(human_angles)
        return self.solve_geometry(points, frames, initial_angles=initial_angles)

    @torch.no_grad()
    def solve_geometry(self, human_points, human_frames, *, initial_angles=None,
                       human_palm_frame=None) -> RetargetingResult:
        """Explicit geometry entry point, also useful for FK consistency checks.

        Uses the same [21,3] landmarks / [5,3,3] frames as DexPilotReward.
        human_palm_frame is only for a consistent rigid re-expression of those
        coordinates; ordinary recorded human poses use the default fixed basis.
        """
        started = time.perf_counter()
        config, hand = self.config, self.hand
        points = torch.as_tensor(human_points, dtype=torch.float64).detach().cpu()
        frames = torch.as_tensor(human_frames, dtype=torch.float64).detach().cpu()
        palm = None if human_palm_frame is None else torch.as_tensor(
            human_palm_frame, dtype=torch.float64).detach().cpu()
        if points.shape != (21, 3) or frames.shape != (5, 3, 3):
            raise ValueError("Expected one human pose: [21,3] points and [5,3,3] frames")
        initial = self.neutral.copy() if initial_angles is None else hand.validate_angles(
            np.asarray(initial_angles, dtype=np.float64)).copy()

        def score(q):
            robot_points, robot_frames = self.geometry(q)
            geometric = self.metric.compare(torch.from_numpy(robot_points), points,
                robot_frames=torch.from_numpy(robot_frames), human_fingertip_frames=frames,
                human_palm_frame=palm)
            cost = geometric.cost + self.regularization_cost(q)
            return RewardOutput(-cost, cost, geometric.squared_vector_errors)

        initial_output = score(initial)
        initial_cost = float(initial_output.cost)
        lower, upper = hand.lower[self.movable], hand.upper[self.movable]

        def expand(x):
            q = self.neutral.copy()
            q[self.movable] = np.clip(x, lower, upper)
            return q

        def objective(x):
            cost = float(score(expand(x)).cost)
            if not np.isfinite(cost):
                raise FloatingPointError("Nonfinite retargeting objective")
            return config.objective_scale * cost

        starts = [initial[self.movable], (lower + upper) / 2]
        rng = np.random.default_rng(config.seed)
        starts.extend(rng.uniform(lower, upper) for _ in range(config.random_starts))
        attempts, candidates = [], []

        def collision_margin(x):
            self.geometry(expand(x))
            penetration = max((c["penetration_m"] for c in self.contacts()), default=0.)
            # A small interior margin avoids accepting an optimizer's slightly
            # infeasible boundary. This is a feasibility constraint, not reward.
            return 1000 * (.5 * config.penetration_tolerance_m - penetration)

        def run(start, *, constrained=False):
            if self.movable.any():
                optimum = minimize(objective, start, method="SLSQP", bounds=list(zip(lower, upper)),
                    constraints=({"type": "ineq", "fun": collision_margin},) if constrained else (),
                    options={"maxiter": config.max_iterations,
                             "ftol": config.collision_ftol if constrained else config.ftol,
                             "eps": config.difference_step})
                q = expand(optimum.x)
                converged, message = bool(optimum.success), str(optimum.message)
                iterations, evaluations = int(optimum.nit), int(optimum.nfev)
            else:
                q, converged, message, iterations, evaluations = self.neutral.copy(), True, "No movable joints", 0, 1
            output = score(q)
            cost = float(output.cost)
            contacts = self.contacts()
            penetration = max((c["penetration_m"] for c in contacts), default=0.)
            accepted = (converged and penetration <= config.penetration_tolerance_m
                        and cost <= initial_cost + 1e-12)
            attempts.append({"converged": converged, "message": message, "iterations": iterations,
                "stage": "collision_refinement" if constrained else "vector_matching",
                "evaluations": evaluations, "cost": cost, "accepted": accepted,
                "geometric_cost": cost - self.regularization_cost(q),
                "regularization_cost": self.regularization_cost(q),
                "max_penetration_m": penetration, "start_angles": expand(start).tolist()})
            candidates.append((q, output.squared_vector_errors.numpy().copy(), contacts))
        for start in starts if self.movable.any() else [np.empty(0)]:
            run(start)
        if (not config.reject_colliding_best and config.collision_refinement
                and self.movable.any() and not any(a["accepted"] for a in attempts)):
            # Keep the original attempts in the report. Refine the least
            # penetrating converged solution; never relax the acceptance bound.
            refinable = [i for i, a in enumerate(attempts) if a["converged"]
                         and a["max_penetration_m"] > config.penetration_tolerance_m]
            if refinable:
                best = min(refinable, key=lambda i: (attempts[i]["max_penetration_m"], attempts[i]["cost"]))
                run(candidates[best][0][self.movable], constrained=True)
                # A deeply folded optimum can trap the nonsmooth contact
                # constraint. Retry the original starts (including neutral)
                # with constraints active from the outset, stopping at the
                # first accepted solution. Keep the same cost and bounds.
                for start in starts:
                    if any(a["accepted"] for a in attempts):
                        break
                    run(start, constrained=True)
        valid = [i for i, attempt in enumerate(attempts) if attempt["accepted"]]
        eligible = ([i for i, a in enumerate(attempts) if a["converged"]]
                    if config.reject_colliding_best else valid)
        selected = min(eligible or range(len(attempts)), key=lambda i: attempts[i]["cost"])
        attempt = attempts[selected]
        q, errors, contacts = candidates[selected]
        accepted = attempt["accepted"]
        status = ("accepted" if accepted else "not_converged" if not attempt["converged"] else
                  "penetration_rejected" if attempt["max_penetration_m"] > config.penetration_tolerance_m
                  else "worse_than_initial")
        # Leave the private FK state consistent with the reported candidate.
        self.geometry(q)
        return RetargetingResult(accepted, status, q, initial_cost, attempt["cost"], errors,
            attempts, selected, contacts, time.perf_counter() - started,
            geometric_cost=attempt["geometric_cost"], regularization_cost=attempt["regularization_cost"],
            initial_geometric_cost=initial_cost - self.regularization_cost(initial),
            initial_regularization_cost=self.regularization_cost(initial), human_scale=self.group.human_scale)
