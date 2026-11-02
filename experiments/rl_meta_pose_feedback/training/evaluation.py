"""Reproduce a saved PPO policy's deterministic validation/test rollouts."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rl_meta_pose_feedback.training.config import TrainingConfig
from rl_meta_pose_feedback.training.runtime import create_runtime, evaluate_policy, select_episodes, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New evaluation directory")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--windows", type=int, help="Held-out window count; defaults to checkpoint validation size")
    parser.add_argument("--selection", type=Path, help="Reuse window_indices from an existing metrics.json")
    parser.add_argument("--emg-control", choices=("matched", "permuted", "zero", "training-mean"), default="matched",
                        help="Diagnostic EMG replacement; references stay fixed")
    parser.add_argument("--zero-actions", action="store_true", help="Hold the reset target without actor commands")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite evaluation output")
    if args.zero_actions and args.emg_control != "matched":
        parser.error("Zero-action control does not need an EMG replacement")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = TrainingConfig.from_dict(checkpoint["config"])
    if args.device is not None:
        config = replace(config, device=args.device)
    actor, _, normalizer, rollout = create_runtime(config, checkpoint)
    indices = checkpoint["pool_indices"].get(args.split)
    if args.selection is not None:
        import json
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        if selection["split"] != args.split:
            parser.error("Frozen evaluation selection belongs to a different split")
        indices = selection["window_indices"]
    count = args.windows if args.windows is not None else len(indices) if indices is not None else config.validation_windows
    if count <= 0:
        parser.error("Evaluation window count must be positive")
    episodes, selected, rejected = select_episodes(config, actor, rollout, args.split,
                                                  count, selected_indices=indices)
    if args.emg_control == "permuted":
        if len(episodes) < 2:
            parser.error("Permuting EMG needs at least two evaluation windows")
        episodes = [replace(episode, sample={**episode.sample, "emg": episodes[(i+1)%len(episodes)].sample["emg"]})
                    for i, episode in enumerate(episodes)]
    elif args.emg_control in ("zero", "training-mean"):
        if args.emg_control == "training-mean" and config.emg_preprocessing == "raw":
            parser.error("Raw-input checkpoint has no fitted training mean; use --emg-control zero")
        constant = (torch.zeros(16) if args.emg_control == "zero" else normalizer.mean.detach().cpu())
        episodes = [replace(episode, sample={**episode.sample,
                    "emg": constant.expand_as(episode.sample["emg"]).clone()})
                    for episode in episodes]
    result, outcomes = evaluate_policy(actor, normalizer, rollout, episodes, zero_actions=args.zero_actions)
    args.output.mkdir(parents=True)
    write_json(args.output / "metrics.json", {"checkpoint": str(args.checkpoint.resolve()),
        "update": checkpoint["update"], "split": args.split, "window_indices": selected,
        "windows": [{"filename": e.sample["filename"], "side": e.sample.get("side", "right"), "segment_id": e.sample["segment_id"],
                     "start": e.sample["start"], "stop": e.sample["stop"]} for e in episodes],
        "emg_preprocessing": config.emg_preprocessing,
        "episode_initialization": config.episode_initialization,
        "reference_hand_convention": "canonical_right_from_unchanged_joint_angles",
        "initializations": [e.initialization for e in episodes],
        "rejected": rejected, "deterministic_mean_actions": not args.zero_actions,
        "temporal_protocol": actor.encoder.config.temporal_protocol, "online_causal": False,
        "actor_inputs": ["EMG_history", "initial_human_pose",
                         "internally_accumulated_command_pose", "morphology"],
        "emg_control": args.emg_control, "zero_actions": args.zero_actions, **result})
    for i, outcome in enumerate(outcomes):
        np.savez_compressed(args.output / f"episode_{i:03d}.npz", **outcome.trace)
    print(f"{args.split} cost={result['mean_motion_cost']:.8f}; traces={args.output}")


if __name__ == "__main__":
    main()
