"""Frozen Meta tracking features with official autoregressive human-pose feedback.

Whole-window linear interpolation is preserved: this is an OFFLINE experiment,
not a causal streaming adapter. Simulator guard samples are excluded.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import torch
from torch import nn
from torch.nn.functional import interpolate
from emg_control_paths import resolve_experiment_path
from .emg2pose_tds import Conv1dBlock, TdsNetwork, TdsStage

TRACKING_SHA256 = "629783b2b1e8bc1316190fc2a57ba95396dd0d6b59cfbb38543fc6ad8bd61ae5"
TEMPORAL_PROTOCOL = "meta_offline_linear_align_corners_v1"


@dataclass(frozen=True)
class EncoderConfig:
    checkpoint: str = "assets/meta/tracking_vemg2pose.ckpt"
    checkpoint_sha256: str = TRACKING_SHA256
    latent_dim: int = 512
    rollout_hz: int = 50
    frozen: bool = True
    temporal_protocol: str = TEMPORAL_PROTOCOL

    def __post_init__(self):
        if (self.latent_dim != 512 or self.rollout_hz != 50 or self.frozen is not True
                or self.checkpoint_sha256 != TRACKING_SHA256
                or self.temporal_protocol != TEMPORAL_PROTOCOL):
            raise ValueError("Requires the frozen official tracking model at 50 Hz")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(resolve_experiment_path(path).read_text(encoding="utf-8")))

    def to_dict(self):
        return asdict(self)


def build_emg2pose_frontend():
    return TdsNetwork(
        conv_blocks=[Conv1dBlock(16, 256, kernel_size=11, stride=5),
                     Conv1dBlock(256, 256, kernel_size=5, stride=2)],
        tds_stages=[
            TdsStage(in_channels=256, in_conv_kernel_width=17, in_conv_stride=4,
                     num_blocks=2, channels=16, feature_width=16, kernel_width=9),
            TdsStage(in_channels=256, in_conv_kernel_width=9, in_conv_stride=2,
                     num_blocks=2, channels=16, feature_width=16, kernel_width=5,
                     out_channels=64),
        ])


class TrackingDecoder(nn.Module):
    """Official SequentialLSTM parameters, with state local to each call.

    Meta revision 5f6f62b, config/pose_module/decoder/lstm.yaml.
    License: experiment/core/resources/licenses/emg2pose_LICENSE.txt.
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(84, 512, 2, batch_first=True)
        self.mlp_out = nn.Sequential(nn.LeakyReLU(), nn.Linear(512, 20))
        self.scale = .01


@dataclass
class EncoderOutput:
    latent: torch.Tensor                     # [B,T,512], top-layer h_t
    sample_indices: torch.Tensor             # nominal command START grid
    valid_mask: torch.Tensor
    predicted_pose: torch.Tensor             # [B,T,20], updated human feedback
    feature_available_indices: torch.Tensor  # last raw input needed by feature

    @property
    def last_latent(self):
        return self.latent[:, -1]


class EMGEncoder(nn.Module):
    frontend_left_context = 1790
    frontend_sample_stride = 80
    control_sample_stride = 40
    right_context = 0
    min_input_samples = 1791

    def __init__(self, config=None, *, load_pretrained=True):
        super().__init__()
        self.config = config or EncoderConfig()
        self.network = build_emg2pose_frontend()
        self.decoder = TrackingDecoder()
        if load_pretrained:
            self.load_official_weights()
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def load_official_weights(self):
        path = resolve_experiment_path(self.config.checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.config.checkpoint_sha256:
            raise ValueError("Meta checkpoint checksum differs from pinned official release")
        # The hash-verified official Lightning file contains OmegaConf metadata.
        # Reuse downloaded dependencies; no network/install during training.
        try:
            import omegaconf  # noqa: F401
        except ImportError as error:
            raise RuntimeError("Install requirements.txt before loading Meta weights") from error
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]
        if any(not key.startswith("model.") for key in state):
            raise ValueError("Unexpected official checkpoint state")
        self.load_state_dict({key.removeprefix("model."): value for key, value in state.items()}, strict=True)

    @torch.no_grad()
    def forward(self, emg, human_q0, *, steps=None):
        if emg.ndim != 3 or emg.shape[0] == 0 or emg.shape[-1] != 16 or not emg.is_floating_point():
            raise ValueError("emg must be floating [B,samples,16]")
        if steps is None:
            steps = round((emg.shape[1] - self.frontend_left_context) / 2000 * 50)
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        count = self.frontend_left_context + self.control_sample_stride * steps
        if emg.shape[1] < count:
            raise ValueError("EMG window does not cover requested 50 Hz sequence")
        emg = emg[:, :count]  # Exclude reward/clock guard tail from interpolation.
        if human_q0.shape != (emg.shape[0], 20) or human_q0.device != emg.device:
            raise ValueError("human_q0 must be [B,20] on the EMG device")
        if not torch.isfinite(emg).all() or not torch.isfinite(human_q0).all():
            raise ValueError("EMG and human initial pose must be finite")
        features = self.network(emg.transpose(1, 2))
        feature_count = features.shape[-1]
        features = interpolate(features, size=steps, mode="linear", align_corners=True)
        state = None
        pose = human_q0.to(dtype=features.dtype)
        hidden, poses = [], []
        for t in range(steps):
            inputs = torch.cat((features[:, :, t], pose), dim=-1)
            out, state = self.decoder.lstm(inputs[:, None], state)
            h = out[:, 0]
            pose = pose + self.decoder.mlp_out(h) * self.decoder.scale
            hidden.append(h)
            poses.append(pose)
        indices = self.frontend_left_context + torch.arange(steps, device=emg.device) * self.control_sample_stride
        positions = torch.linspace(0, feature_count - 1, steps, device=emg.device, dtype=torch.float64)
        available = self.frontend_left_context + positions.ceil().long() * self.frontend_sample_stride
        return EncoderOutput(torch.stack(hidden, 1), indices,
                             torch.ones(emg.shape[0], steps, dtype=torch.bool, device=emg.device),
                             torch.stack(poses, 1), available)
