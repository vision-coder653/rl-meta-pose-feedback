"""EMG decoder with optional Gaussian exploration after the synergy map.

The caller supplies the morphology representation, MLP capacity and B
initialization. Target accumulation and physical execution belong to the
environment. Exploration amplitude must be supplied explicitly; a configured
MotorNoise learns one global log standard deviation per motor by default.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from configs.groups import TrainingGroupConfig

from .action_distribution import MaskedMotorNormal, MotorNoise
from .command_head import CommandHead, expand_condition
from .encoder import EMGEncoder, EncoderConfig, EncoderOutput
from .synergy import Synergy


@dataclass(frozen=True)
class DecoderConfig:
    fusion: str = "hidden_morph_command_v2"
    morphology_dim: int = 35
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: str = "gelu"

    def __post_init__(self):
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        if (self.fusion != "hidden_morph_command_v2" or self.morphology_dim != 35
                or any(type(size) is not int or size <= 0 for size in self.hidden_sizes)
                or self.activation not in ("relu", "gelu", "tanh")):
            raise ValueError("Invalid decoder input dimension, hidden widths or activation")

    @classmethod
    def load(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class CommandOutput:
    coefficients: torch.Tensor
    motor_increment_mean: torch.Tensor
    motor_mask: torch.Tensor

    def distribution(self, *, std: float | torch.Tensor) -> MaskedMotorNormal:
        """Build the approved motor-space Gaussian with an explicit std in rad."""
        return MaskedMotorNormal(self.motor_increment_mean, self.motor_mask, std=std)


@dataclass
class DecoderOutput:
    encoder: EncoderOutput
    command: CommandOutput

    @property
    def last_increment(self) -> torch.Tensor:
        last = self.encoder.valid_mask.sum(dim=1) - 1
        batch = torch.arange(last.shape[0], device=last.device)
        return self.command.motor_increment_mean[batch, last]


class EMGDecoder(nn.Module):
    def __init__(self, encoder: EMGEncoder, command_head: CommandHead, synergy: Synergy, *,
                 motor_noise: MotorNoise | None = None,
                 training_group: TrainingGroupConfig | None = None):
        super().__init__()
        if encoder.config.latent_dim != command_head.latent_dim:
            raise ValueError("Encoder and command head latent dimensions differ")
        self.encoder, self.command_head, self.synergy = encoder, command_head, synergy
        self.motor_noise = motor_noise
        self.training_group = training_group

    @classmethod
    def from_config(cls, decoder_config: DecoderConfig, encoder_config: EncoderConfig, *,
                    motor_noise: MotorNoise | None = None,
                    training_group: TrainingGroupConfig | None = None, load_pretrained: bool = True):
        return cls(
            EMGEncoder(encoder_config, load_pretrained=load_pretrained),
            CommandHead(encoder_config.latent_dim, decoder_config.morphology_dim,
                        hidden_sizes=decoder_config.hidden_sizes, activation=decoder_config.activation),
            Synergy.orthogonal(),
            motor_noise=motor_noise,
            training_group=training_group,
        )

    def get_extra_state(self):
        """Keep correspondence/scale/frame metadata with grouped model weights."""
        return {"training_group": None if self.training_group is None else self.training_group.to_dict(),
                "actor_inputs": "frozen_hidden_morphology_masked_pre_action_command_pose_v2",
                "fusion": "hidden_morph_command_v2"}

    def set_extra_state(self, state):
        expected = self.get_extra_state()
        if state != expected:
            raise RuntimeError("Checkpoint training group differs; use the same group and reward configuration")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Preserve loading of older, ungrouped interface-check weights. A grouped
        # controller must never accept a checkpoint with missing group metadata.
        key = prefix + "_extra_state"
        if key not in state_dict and self.training_group is None:
            state_dict[key] = self.get_extra_state()
        elif key not in state_dict:
            error_msgs.append("Grouped decoder checkpoint is missing training-group metadata")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def action_distribution(self, command: CommandOutput) -> MaskedMotorNormal:
        """Use configured noise; its parameters are included in decoder.parameters()."""
        if self.motor_noise is None:
            raise RuntimeError("Provide MotorNoise with an explicit initial_std before sampling")
        return command.distribution(std=self.motor_noise.stddev)

    def decode_latent(self, latent: torch.Tensor, morphology: torch.Tensor,
                      motor_mask: torch.Tensor,
                      valid_tokens: torch.Tensor | None = None, *,
                      command_pose: torch.Tensor) -> CommandOutput:
        mask = expand_condition(motor_mask, latent, 20, "motor_mask")
        if mask.dtype != torch.bool:
            raise ValueError("motor_mask must be boolean")
        descriptor = expand_condition(morphology, latent, 35, "morphology")
        per_finger = descriptor[..., 5:].reshape(*latent.shape[:-1], 5, 6)
        presence = torch.cat((descriptor[..., :5, None], per_finger[..., :3]), dim=-1)
        if not torch.equal(presence.reshape_as(mask).bool(), mask):
            raise ValueError("Morphology joint presence differs from motor_mask")
        coefficients = self.command_head(latent, morphology, command_pose=command_pose, motor_mask=mask)
        if valid_tokens is not None:
            if valid_tokens.shape != latent.shape[:-1] or valid_tokens.dtype != torch.bool:
                raise ValueError("valid_tokens must be a boolean mask matching latent batch/time dimensions")
        if self.training_group is not None:
            expected = torch.zeros(5, dtype=torch.bool, device=mask.device)
            expected[[finger - 1 for finger in self.training_group.robot_finger_ids]] = True
            matches = (mask.reshape(*mask.shape[:-1], 5, 4).any(dim=-1) == expected).all(dim=-1)
            if valid_tokens is not None:
                matches = matches | ~valid_tokens
            if not matches.all():
                raise ValueError("Motor mask finger IDs disagree with the decoder's training group")
        if valid_tokens is not None:
            coefficients = coefficients.masked_fill(~valid_tokens[..., None], 0)
            mask = mask & valid_tokens[..., None]
        mean = self.synergy(coefficients, mask)
        return CommandOutput(coefficients, mean, mask)

    def forward(self, emg: torch.Tensor, morphology: torch.Tensor, motor_mask: torch.Tensor,
                *, human_q0: torch.Tensor, steps: int, command_pose: torch.Tensor) -> DecoderOutput:
        encoded = self.encoder(emg, human_q0, steps=steps)
        command = self.decode_latent(encoded.latent, morphology, motor_mask, encoded.valid_mask,
                                     command_pose=command_pose)
        return DecoderOutput(encoded, command)
