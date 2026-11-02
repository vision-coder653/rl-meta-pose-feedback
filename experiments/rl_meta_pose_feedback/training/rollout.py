"""Continuous control from frozen Meta tracking features and initial human pose.

Raw source segments remain intact. References are looked up at step END times;
only the requested FK frames are computed. Episodes are finite reference tasks,
with a terminal final state. No frame is teacher-forced or reset within a clip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from simulation.environment import HandEnvironment, HandState
from configs.groups import TrainingGroupConfig
from kinematics import HumanReference, RobotLandmarks
from rl_meta_pose_feedback.models.action_distribution import MaskedMotorNormal
from rl_meta_pose_feedback.models.policy import EMGDecoder
from rl_meta_pose_feedback.models.value import EMGNormalizer
from simulation.reward import DexPilotReward, open_pose_cost
from simulation.retargeting import PoseRetargeter
from rl_meta_pose_feedback.training.config import TrainingConfig

CRITIC_FEATURE_VERSION = "robot_q_qdot_target_human_angles_morph_remaining_v1"
CRITIC_INPUT_DIM = 20 * 4 + 35 + 1
DATA_PREPARATION_VERSION = 7  # New runs track geometry without the open-pose prior.


class ReferenceCoverageError(ValueError):
    """A sample is too short in recorded time to cover the finite task."""


REFERENCE_COVERAGE_ERROR = "Window lacks time coverage for the step-end reference"
RETARGET_COLLISION_ERROR = "Best retarget pose exceeds penetration tolerance"


class RetargetCollisionError(RuntimeError):
    """A collision-rejected initial match; samplers may skip this exact window."""

    def __init__(self, result, sample, control_time):
        super().__init__(f"{RETARGET_COLLISION_ERROR}: filename={sample.get('filename')}, "
                         f"start={sample.get('start')}, status={result.status}")
        self.details = {"retargeting": result.to_dict(), "control_time": control_time,
                        "filename": sample.get("filename"), "start": sample.get("start"),
                        "stop": sample.get("stop")}


def critic_observation(state: HandState, human_angles: torch.Tensor, morphology: np.ndarray,
                       *, remaining: float, velocity_scale: float) -> np.ndarray:
    """Privileged PRE-action information; never passed to the actor."""
    result = np.concatenate((state.joint_angles / math.pi,
                             np.clip(state.joint_velocities / velocity_scale, -10, 10),
                             state.target_angles / math.pi,
                             np.asarray(human_angles.cpu()) / math.pi, morphology, [remaining])).astype(np.float32)
    if result.shape != (CRITIC_INPUT_DIM,) or not np.isfinite(result).all():
        raise ValueError("Invalid critic observation")
    return result


@dataclass
class PreparedEpisode:
    sample: dict
    reference_angles: torch.Tensor
    human_points: torch.Tensor
    human_frames: torch.Tensor
    query_times: torch.Tensor
    sampled_times: torch.Tensor
    initial_angles: np.ndarray | None
    initialization: dict
    human_initial_pose: torch.Tensor


@dataclass
class EpisodeResult:
    critic_observations: torch.Tensor
    costs: torch.Tensor
    clipped_fraction: float
    trace: dict


class FixedHandRollout:
    def __init__(self, config: TrainingConfig, environment: HandEnvironment,
                 landmarks: RobotLandmarks, group: TrainingGroupConfig):
        self.config, self.environment, self.landmarks = config, environment, landmarks
        group = config.reward_group(group)
        self.group, self.metric = group, DexPilotReward(group).float().eval()
        self.reward_settings = config.reward_settings(group)
        self.neutral = np.clip(np.zeros(20), environment.hand.lower, environment.hand.upper)
        self.movable = environment.hand.active_mask & (environment.hand.upper > environment.hand.lower)
        self.morphology = environment.hand.policy_features(finger_tokens=config.finger_tokens)
        self.mask = environment.hand.active_mask.copy()
        self.initial_angles = environment.hand.validate_angles(np.array(config.initial_angles))
        self.period = environment.physics_timestep * config.physics_steps
        if not np.isclose(self.period, .02, rtol=0, atol=1e-10):
            raise ValueError("Meta tracking requires a 20 ms command period")
        self.palm_finger = group.robot_palm_reference_finger
        if landmarks.finger_ids != group.robot_finger_ids:
            raise ValueError("Candidate fingers do not match the training group")
        landmarks.palm_rest_frame(reference_finger=self.palm_finger)
        self.retargeter = (PoseRetargeter(environment.hand, landmarks, group, config.retargeting)
                           if config.episode_initialization == "retarget" else None)

    @torch.no_grad()
    def prepare(self, sample: dict, *, left_context: int, stride: int,
                initialize: bool = True) -> PreparedEpisode:
        steps = self.config.episode_steps
        if not np.isclose(self.period, stride / self.config.nominal_sample_rate, atol=1e-10, rtol=0):
            raise ValueError("Physics command period must match the 50 Hz control grid")
        time = sample["time"]
        # Continuity was indexed on the raw sequence, not on downsampled labels.
        reference = HumanReference(time, sample["joint_angles"], max_gap_seconds=sample["max_gap_seconds"])
        indices = left_context + stride * torch.arange(steps + 1)
        if indices[-1] >= len(time):
            raise ValueError("Window lacks the step-end reference and guard samples")
        query = time[indices[0]] + torch.arange(steps + 1, dtype=torch.float64) * self.period
        if query[-1] > time[-1]:
            raise ReferenceCoverageError(REFERENCE_COVERAGE_ERROR)
        # Raw continuity already bounds nearest-sample error. Do not reject a
        # continuous window for drift from the nominal 2 kHz / 25 Hz clock, or
        # apply a second, stricter timestamp tolerance to its reference labels.
        selected = reference.sample(query, max_error_seconds=sample["max_gap_seconds"])
        angles = selected.angles.float().cpu()
        initial_angles = None
        initialization = {"mode": self.config.episode_initialization,
                          "reference_query_time": float(query[0]),
                          "reference_sample_time": float(selected.time[0])}
        episode = PreparedEpisode(sample, angles, self.metric.human_fk(angles),
                                  self.metric.human_fk.fingertip_frames(angles), query, selected.time,
                                  initial_angles, initialization, angles[0].clone())
        return self.initialize_episode(episode) if initialize else episode

    def initialize_episode(self, episode, result=None):
        """Apply the identical acceptance checks to local or worker solver results."""
        sample, initialization = episode.sample, episode.initialization
        initial_angles = self.initial_angles.copy()
        if self.retargeter is not None:
            # The first control reference follows the encoder's history.
            # Solve once per prepared window; repeated rollouts reuse it.
            if result is None:
                result = self.retargeter.solve(episode.human_initial_pose, initial_angles=initial_angles)
            initialization["retargeting"] = result.to_dict()
            if not result.accepted:
                if result.status == "penetration_rejected":
                    raise RetargetCollisionError(result, sample, float(episode.query_times[0]))
                raise RuntimeError(
                    f"Episode alignment failed: filename={sample.get('filename')}, "
                    f"start={sample.get('start')}, control_time={float(episode.query_times[0])}, "
                    f"status={result.status}, cost={result.cost}; no fixed-pose fallback")
            initial_angles = result.require_angles()
        initialization["joint_angles"] = initial_angles.tolist()
        episode.initial_angles = initial_angles
        return episode

    @torch.no_grad()
    def encode(self, actor, normalizer, episodes):
        device = next(actor.parameters()).device
        emg = torch.stack([episode.sample["emg"] for episode in episodes]).to(device)
        q0 = torch.stack([episode.human_initial_pose for episode in episodes]).to(device)
        return actor.encoder(normalizer(emg), q0, steps=self.config.episode_steps)

    @torch.no_grad()
    def actions(self, actor, normalizer, episodes, *, stochastic, encoded=None):
        # Commands use internal issued-target history, never measured robot q.
        encoded = self.encode(actor, normalizer, episodes) if encoded is None else encoded
        device = encoded.latent.device
        mask = torch.as_tensor(self.mask, device=device)
        current = np.where(self.mask, np.stack([e.initial_angles for e in episodes]).astype(np.float64), 0.)
        actions, log_probs = [], []
        for t in range(self.config.episode_steps):
            command = actor.decode_latent(encoded.latent[:, t], torch.as_tensor(self.morphology, device=device),
                mask, encoded.valid_mask[:, t],
                command_pose=torch.as_tensor(current, device=device, dtype=encoded.latent.dtype))
            distribution = MaskedMotorNormal(command.motor_increment_mean, mask, std=actor.motor_noise.stddev)
            action = distribution.sample() if stochastic else command.motor_increment_mean
            actions.append(action.cpu())
            log_probs.append(distribution.log_prob(action).cpu())
            current = self.advance_command(current, action.cpu().numpy())
        return torch.stack(actions, 1), torch.stack(log_probs, 1)

    def advance_command(self, current, raw_action):
        """Match the device's commanded-target accumulator, without physics."""
        delta = np.where(self.mask, np.asarray(raw_action, dtype=np.float64), 0.)
        limit = self.environment.max_joint_increment
        if limit is not None:
            delta = np.clip(delta, -limit, limit)
        hand = self.environment.hand
        return np.where(self.mask, np.clip(current + delta, hand.lower, hand.upper), 0.)

    def command_history(self, episodes, raw_actions):
        """PRE-action observations reconstructed from recorded sampled actions.

        PPO conditions on this fixed history, not a trajectory generated with
        its newly updated policy. No actual joint positions are consulted.
        """
        current = np.where(self.mask, np.stack([e.initial_angles for e in episodes]).astype(np.float64), 0.)
        history = []
        for actions in raw_actions.detach().cpu().numpy().transpose(1, 0, 2):
            history.append(current.copy())
            current = self.advance_command(current, actions)
        return torch.from_numpy(np.stack(history, 1))

    @torch.no_grad()
    def execute(self, episode: PreparedEpisode, actions: torch.Tensor) -> EpisodeResult:
        steps, config = self.config.episode_steps, self.config
        if actions.shape != (steps, 20) or not torch.isfinite(actions).all():
            raise ValueError("Expected finite raw actions [episode_steps,20]")
        if episode.initial_angles is None:
            raise ValueError("Episode was prepared for saved-trace replay only; initialization is required")
        state = self.environment.reset(episode.initial_angles)
        initial_angles = state.joint_angles.copy()
        observations, points, frames, actual, targets, clipped = [], [], [], [], [], []
        for t, action in enumerate(actions.numpy()):
            observations.append(critic_observation(state, episode.reference_angles[t], self.morphology,
                                                   remaining=(steps-t)/steps,
                                                   velocity_scale=config.critic_velocity_scale))
            step = self.environment.step(action, physics_steps=config.physics_steps)
            state = step.state
            if abs(state.time - (t+1)*self.period) > 1e-8:
                raise RuntimeError("Unexpected simulation time during an episode")
            points.append(self.landmarks.positions(state))
            frames.append(self.landmarks.orientations(state, palm_finger=self.palm_finger))
            actual.append(state.joint_angles)
            targets.append(state.target_angles)
            clipped.append(step.clipped_slots[self.mask].mean())
        output = self.metric.compare(torch.tensor(np.asarray(points), dtype=torch.float32),
                                     episode.human_points[1:],
                                     robot_frames=torch.tensor(np.asarray(frames), dtype=torch.float32),
                                     human_fingertip_frames=episode.human_frames[1:])
        costs = output.cost
        regularization = torch.zeros_like(costs)
        if self.reward_settings["open_regularization"]:
            # Historical checkpoints retain their original evaluation objective.
            regularization = torch.as_tensor(open_pose_cost(
                actual, self.neutral, self.movable, self.reward_settings["open_regularization"]),
                dtype=costs.dtype, device=costs.device)
            costs = costs + regularization
        if not torch.isfinite(costs).all():
            raise RuntimeError("Nonfinite reference motion cost")
        trace = {"simulation_time": (np.arange(steps)+1)*self.period,
                 "command_pose_before": np.vstack((episode.initial_angles, np.asarray(targets)[:-1])),
                 "initial_joint_angles": initial_angles,
                 "initial_human_angles": episode.human_initial_pose.numpy(),
                 "reference_query_time": episode.query_times[1:].numpy(),
                 "reference_sample_time": episode.sampled_times[1:].numpy(),
                 "actual_joint_angles": np.asarray(actual), "target_angles": np.asarray(targets),
                 "raw_actions": actions.numpy(), "costs": costs.numpy(),
                 "geometric_costs": output.cost.numpy(), "regularization_costs": regularization.numpy(),
                 "squared_vector_errors": output.squared_vector_errors.numpy()}
        return EpisodeResult(torch.from_numpy(np.asarray(observations)), costs,
                             float(np.mean(clipped)), trace)
