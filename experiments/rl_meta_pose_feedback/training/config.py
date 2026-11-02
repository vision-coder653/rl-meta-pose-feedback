"""Reproducible fixed-hand PPO settings; paths resolve from experiment/."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from simulation.retargeting import RetargetingConfig
from emg_control_paths import EXPERIMENT_DIR, resolve_experiment_path


@dataclass(frozen=True)
class PPOConfig:
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    adam_epsilon: float = 1e-5
    gamma: float = math.sqrt(.99)
    gae_lambda: float = math.sqrt(.95)
    clip_ratio: float = .2
    target_kl: float = .02
    kl_control: str = "adaptive_lr"
    actor_min_learning_rate: float = 1e-6
    actor_max_learning_rate: float = 1e-3
    kl_lr_factor: float = 1.5
    update_epochs: int = 5
    minibatch_episodes: int = 2
    max_grad_norm: float = .5
    value_coefficient: float = .5
    entropy_coefficient: float = 0.

    def __post_init__(self):
        positive = ("actor_learning_rate", "critic_learning_rate", "adam_epsilon",
                    "clip_ratio", "target_kl", "max_grad_norm", "value_coefficient",
                    "actor_min_learning_rate", "actor_max_learning_rate", "kl_lr_factor")
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.kl_control not in ("adaptive_lr", "early_stop"):
            raise ValueError("kl_control must be adaptive_lr or early_stop")
        if self.kl_lr_factor <= 1 or self.actor_min_learning_rate > self.actor_max_learning_rate:
            raise ValueError("Invalid adaptive KL learning-rate bounds or factor")
        if self.kl_control == "adaptive_lr" and not (
                self.actor_min_learning_rate <= self.actor_learning_rate <= self.actor_max_learning_rate):
            raise ValueError("Initial actor learning rate must be within adaptive bounds")
        for name in ("gamma", "gae_lambda", "entropy_coefficient"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.gamma > 1 or self.gae_lambda > 1 or self.clip_ratio >= 1:
            raise ValueError("gamma/lambda must be <=1 and clip_ratio <1")
        for name in ("update_epochs", "minibatch_episodes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TrainingConfig:
    schema_version: int = 1
    seed: int = 0
    device: str = "auto"
    mjcf: str = "resources/robot/hand_f1_990_f2_990_f3_990_f4_990_f5_990_complete.xml"
    group: str = "anth21"
    group_config: str = "configs/training_groups.json"
    encoder_config: str = "experiments/rl_meta_pose_feedback/configs/encoder.json"
    decoder_config: str = "experiments/rl_meta_pose_feedback/configs/decoder.json"
    environment_config: str = "experiments/rl_meta_pose_feedback/configs/environment.json"
    data_dir: str = "data/emg2pose_data"
    manifest: str = "data/both_split.json"
    interval_index: str = "data/both_intervals.jsonl"
    finger_tokens: tuple[str, ...] | None = None
    initial_angles: tuple[float, ...] = (0.,) * 20
    episode_initialization: str = "retarget"
    reward_objective: str = "tracking"
    retargeting: RetargetingConfig = field(default_factory=RetargetingConfig)
    physics_steps: int = 10
    nominal_sample_rate: float = 2000.
    # Accepted only for reading old configurations; no longer filters windows.
    reference_tolerance_seconds: float | None = None
    episode_steps: int = 256
    episodes_per_update: int = 8
    total_updates: int | None = 20  # None runs until validation early stopping.
    train_windows: int = 64
    validation_windows: int = 8
    training_sampling: str = "fixed_pool"
    training_jitter: bool = True
    emg_preprocessing: str = "raw"
    normalization_windows: int = 32
    normalization_clip: float = 10.
    initial_std: float = .02 / math.sqrt(2)
    actor_output_gain: float = .005
    encoder_dropout: float = 0.
    reward_scale: float = .5
    critic_hidden_sizes: tuple[int, ...] = (128, 128)
    critic_velocity_scale: float = 10.
    evaluate_every: int = 5
    checkpoint_every: int = 1
    early_stop_patience: int = 0
    early_stop_min_updates: int = 100
    early_stop_relative_improvement: float = .005
    torch_threads: int = 2
    retarget_workers: int = 0  # Zero preserves serial execution for old configs.
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self):
        if self.emg_preprocessing != "raw" or self.nominal_sample_rate != 2000.:
            raise ValueError("Frozen Meta weights require raw 2 kHz EMG")
        if self.physics_steps != 10:
            raise ValueError("This experiment requires 10 physics substeps at 2 ms (50 Hz)")
        if self.reward_objective not in ("tracking", "retarget", "legacy_vector"):
            raise ValueError("reward_objective must be tracking, retarget or legacy_vector")
        if self.episode_initialization not in ("retarget", "fixed"):
            raise ValueError("episode_initialization must be retarget or fixed")
        if not isinstance(self.retargeting, RetargetingConfig):
            raise ValueError("retargeting must be a RetargetingConfig")
        if self.schema_version != 1 or type(self.seed) is not int:
            raise ValueError("Unsupported configuration version or seed")
        if self.device not in ("auto", "cpu", "cuda") or self.encoder_dropout != 0:
            raise ValueError("Use auto/cpu/cuda and zero encoder dropout for PPO likelihood replay")
        if self.training_sampling not in ("fixed_pool", "stream"):
            raise ValueError("training_sampling must be fixed_pool or stream")
        if type(self.training_jitter) is not bool:
            raise ValueError("training_jitter must be a boolean")
        if self.emg_preprocessing not in ("raw", "standardize"):
            raise ValueError("emg_preprocessing must be raw or standardize")
        for name in ("early_stop_patience", "early_stop_min_updates", "retarget_workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        improvement = self.early_stop_relative_improvement
        if isinstance(improvement, bool) or not isinstance(improvement, (float, int)) or not math.isfinite(improvement) or not 0 <= improvement < 1:
            raise ValueError("early_stop_relative_improvement must be finite and in [0,1)")
        if self.total_updates is None:
            if not self.early_stop_patience:
                raise ValueError("Unlimited updates require validation early stopping")
        elif type(self.total_updates) is not int or self.total_updates <= 0:
            raise ValueError("total_updates must be a positive integer or null")
        for name in ("physics_steps", "episode_steps", "episodes_per_update",
                     "train_windows", "validation_windows", "normalization_windows", "evaluate_every",
                     "checkpoint_every", "torch_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.reference_tolerance_seconds is not None:
            value = self.reference_tolerance_seconds
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Legacy reference_tolerance_seconds must be finite and positive")
        for name in ("nominal_sample_rate", "normalization_clip",
                     "initial_std", "actor_output_gain", "reward_scale", "critic_velocity_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "initial_angles", tuple(self.initial_angles))
        if len(self.initial_angles) != 20 or any(not math.isfinite(x) for x in self.initial_angles):
            raise ValueError("initial_angles must contain 20 finite radians")
        object.__setattr__(self, "critic_hidden_sizes", tuple(self.critic_hidden_sizes))
        if not self.critic_hidden_sizes or any(type(x) is not int or x <= 0 for x in self.critic_hidden_sizes):
            raise ValueError("critic_hidden_sizes must contain positive integers")
        if self.finger_tokens is not None:
            object.__setattr__(self, "finger_tokens", tuple(self.finger_tokens))
        if not isinstance(self.ppo, PPOConfig):
            raise ValueError("ppo must be a PPOConfig")

    def reward_group(self, group):
        return self.retargeting.resolve_group(group) if self.reward_objective in ("tracking", "retarget") else group

    def reward_settings(self, group) -> dict:
        # Keep the old combined objective only for historical checkpoint replay.
        # New tracking rewards share retargeting's scale, not its open-pose prior.
        return {"definition": self.reward_objective,
                "human_scale": self.reward_group(group).human_scale,
                "open_regularization": self.retargeting.open_regularization
                    if self.reward_objective == "retarget" else 0.}

    def path(self, name: str) -> Path:
        return resolve_experiment_path(getattr(self, name))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict):
        value = dict(value)
        # Old checkpoints always standardized inputs; never reinterpret their weights.
        value.setdefault("emg_preprocessing", "standardize")
        # Preserve historical sampling when loading configurations/checkpoints.
        value.setdefault("training_jitter", False)
        # Missing fields identify historical configurations/checkpoints.
        value.setdefault("episode_initialization", "fixed")
        # Historical weights keep their original unregularized vector objective.
        value.setdefault("reward_objective", "legacy_vector")
        value["retargeting"] = RetargetingConfig(**value.get("retargeting", {}))
        ppo = dict(value.get("ppo", {}))
        # Historical checkpoint/config files retain their original algorithm.
        ppo.setdefault("kl_control", "early_stop")
        value["ppo"] = PPOConfig(**ppo)
        return cls(**value)

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
