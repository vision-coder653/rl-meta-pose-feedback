"""Shared model/environment assembly and deterministic policy evaluation.

Training, independent evaluation and replay use the same runtime definitions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from configs.groups import load_training_groups
from kinematics import RobotLandmarks
from rl_meta_pose_feedback.models.action_distribution import MotorNoise
from rl_meta_pose_feedback.models.encoder import EncoderConfig
from rl_meta_pose_feedback.models.policy import DecoderConfig, EMGDecoder
from rl_meta_pose_feedback.models.value import EMGNormalizer, ValueNetwork
from simulation.environment import HandEnvironment
from simulation.morphology import HandMorphology
from rl_meta_pose_feedback.training.config import TrainingConfig
from rl_meta_pose_feedback.training.rollout import CRITIC_FEATURE_VERSION, CRITIC_INPUT_DIM, FixedHandRollout
from rl_meta_pose_feedback.training.sampling import EpisodeSampler


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def create_runtime(config: TrainingConfig, checkpoint=None):
    torch.set_num_threads(config.torch_threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
    hand = HandMorphology.load(config.path("mjcf"))
    group = load_training_groups(config.path("group_config"))[config.group]
    group = config.reward_group(group)
    reward_settings = config.reward_settings(group)
    encoder_config = EncoderConfig.load(config.path("encoder_config"))
    decoder_config = DecoderConfig.load(config.path("decoder_config"))
    if checkpoint is not None:
        if checkpoint.get("experiment_id") != "rl_meta_pose_feedback_frozen_v1":
            raise ValueError("Checkpoint belongs to a different experiment")
        if checkpoint.get("decoder_config") != asdict(decoder_config):
            raise ValueError("Checkpoint decoder fusion differs")
        saved_reward = checkpoint.get("reward_objective")
        if ((saved_reward is None and config.reward_objective != "legacy_vector")
                or (saved_reward is not None and saved_reward != reward_settings)):
            raise ValueError("Checkpoint reward objective differs; do not reinterpret historical weights")
        if (checkpoint["schema_version"] != 1 or checkpoint["critic_feature_version"] != CRITIC_FEATURE_VERSION
                or checkpoint["group"] != group.to_dict()
                or checkpoint["encoder_config"] != encoder_config.to_dict()
                or checkpoint["model_sha256"] != hashlib.sha256(config.path("mjcf").read_bytes()).hexdigest()):
            raise ValueError("Checkpoint geometry, group, encoder or feature definition differs")
    actor = EMGDecoder.from_config(decoder_config, encoder_config,
                                   motor_noise=MotorNoise(config.initial_std), training_group=group,
                                   load_pretrained=checkpoint is None).to(device).eval()
    # Small output initialization is specific to fresh training; old weights are preserved on load.
    if checkpoint is None:
        nn.init.orthogonal_(actor.command_head.network[-1].weight, gain=config.actor_output_gain)
        nn.init.zeros_(actor.command_head.network[-1].bias)
    normalizer = EMGNormalizer(clip=config.normalization_clip, mode=config.emg_preprocessing).to(device)
    critic = ValueNetwork(CRITIC_INPUT_DIM, config.critic_hidden_sizes).to(device)
    environment = HandEnvironment(hand, **json.loads(config.path("environment_config").read_text(encoding="utf-8")))
    rollout = FixedHandRollout(config, environment, RobotLandmarks.from_hod(hand), group)
    if checkpoint is not None:
        actor.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["critic"])
        normalizer.load_state_dict(checkpoint["normalizer"])
    return actor, critic, normalizer, rollout


def select_episodes(config, actor, rollout, split, count, *, selected_indices=None, preparation_pool=None):
    sampler = EpisodeSampler.from_config(config, actor.encoder, rollout, split, preparation_pool=preparation_pool)
    episodes, accepted, rejected = sampler.draw(count, selected_indices=selected_indices)
    print(f"Selected {split}: {len(episodes)} windows, {len(rejected)} skipped windows (reference coverage or retarget collision)", flush=True)
    return episodes, accepted, rejected


@torch.no_grad()
def evaluate_policy(actor, normalizer, rollout, episodes, *, zero_actions=False):
    actor.eval()
    # Bound GPU memory independently of the number of held-out windows.
    batch_size = rollout.config.episodes_per_update
    actions = (torch.zeros(len(episodes), rollout.config.episode_steps, 20) if zero_actions else
               torch.cat([rollout.actions(actor, normalizer, episodes[start:start+batch_size], stochastic=False)[0]
                          for start in range(0, len(episodes), batch_size)]))
    outcomes = [rollout.execute(episode, action) for episode, action in zip(episodes, actions)]
    costs = torch.stack([result.costs for result in outcomes])
    geometric = np.stack([result.trace["geometric_costs"] for result in outcomes])
    regularization = np.stack([result.trace["regularization_costs"] for result in outcomes])
    return {"mean_motion_cost": float(costs.mean()), "per_episode_cost": costs.mean(-1).tolist(),
            "reward_objective": rollout.reward_settings,
            "mean_geometric_cost": float(geometric.mean()),
            "mean_regularization_cost": float(regularization.mean()),
            "local_vector_rms_mm": float(np.sqrt(2*geometric.mean())*1000),
            "per_step_cost": costs.mean(0).tolist(),
            "first_quarter_cost": float(costs[:, :max(1, costs.shape[1]//4)].mean()),
            "last_quarter_cost": float(costs[:, -max(1, costs.shape[1]//4):].mean()),
            "action_clipped_fraction": float(np.mean([item.clipped_fraction for item in outcomes]))}, outcomes
