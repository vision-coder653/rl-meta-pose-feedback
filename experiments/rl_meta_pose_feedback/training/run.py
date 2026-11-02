"""Train a HOD controller on frozen Meta EMG + initial-human-pose features."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from itertools import count
from dataclasses import replace, asdict
from pathlib import Path

import numpy as np
import torch

from rl_meta_pose_feedback.training.sampling import EpisodeSampler
from rl_meta_pose_feedback.training.parallel import RetargetPool
from rl_meta_pose_feedback.training.config import EXPERIMENT_DIR, TrainingConfig
from rl_meta_pose_feedback.training.rollout import CRITIC_FEATURE_VERSION, DATA_PREPARATION_VERSION
from rl_meta_pose_feedback.training.trainer import PPOTrainer, ValidationEarlyStopping
from rl_meta_pose_feedback.training.runtime import create_runtime, evaluate_policy, select_episodes, write_json
from rl_meta_pose_feedback.models.policy import DecoderConfig


def save_checkpoint(path, config, actor, critic, normalizer, rollout, trainer, *, update, pools, data_sampler=None):
    state = {"experiment_id": "rl_meta_pose_feedback_frozen_v1", "schema_version": 1, "data_preparation_version": DATA_PREPARATION_VERSION,
             "config": config.to_dict(), "group": rollout.group.to_dict(),
             "reward_objective": rollout.reward_settings,
             "critic_feature_version": CRITIC_FEATURE_VERSION,
             "encoder_config": actor.encoder.config.to_dict(),
             "decoder_config": asdict(DecoderConfig.load(config.path("decoder_config"))),
             "model_sha256": hashlib.sha256(config.path("mjcf").read_bytes()).hexdigest(),
             "actor": actor.state_dict(), "critic": critic.state_dict(), "normalizer": normalizer.state_dict(),
             "actor_optimizer": trainer.actor_optimizer.state_dict(),
             "critic_optimizer": trainer.critic_optimizer.state_dict(), "update": update,
             "pool_indices": pools, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "trainer_rng": trainer.rng.bit_generator.state}
    if data_sampler is not None:
        state["data_sampler"] = data_sampler.state_dict()
    temporary = Path(path).with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_update_log(path: Path, update: int):
    """Keep only checkpointed updates, preserving any interrupted tail separately."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    committed = [row for row in rows if row["update"] <= update]
    if [r["update"] for r in committed] != list(range(1, update+1)):
        raise ValueError(f"Missing or duplicate committed updates in {path.name}")
    if len(committed) != len(rows):
        backup = path.with_name(f"{path.stem}.interrupted_{time.time_ns()}{path.suffix}")
        shutil.copy2(path, backup)
        path.write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in committed), encoding="utf-8")
    return committed


def run_training(config: TrainingConfig, output: Path, *, resume: Path | None = None):
    with RetargetPool(config) as preparation_pool:
        return _run_training(config, output, resume=resume, preparation_pool=preparation_pool)


def _run_training(config, output, *, resume, preparation_pool):
    output = output.resolve()
    started = time.perf_counter()
    checkpoint, elapsed_before, completed = None, 0., 0
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", config.to_dict())
    else:
        resume = resume.resolve()
        if resume.parent != output or resume.name != "last.pt" or config.training_sampling != "stream":
            raise ValueError("Resume requires this output directory's last.pt and stream sampling")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get("data_preparation_version") != DATA_PREPARATION_VERSION:
            raise ValueError("Data preparation changed (reference filtering, initialization or reward); start a new run")
        old = TrainingConfig.from_dict(checkpoint["config"]).to_dict()
        new = config.to_dict()
        allowed = {"total_updates", "device", "torch_threads", "checkpoint_every", "retarget_workers",
                   "early_stop_patience", "early_stop_min_updates", "early_stop_relative_improvement"}
        if any(old[k] != new[k] for k in old.keys()-allowed):
            raise ValueError("Resume may change budget/early stopping, not data, architecture, reward or PPO settings")
        completed = checkpoint["update"]
        if config.total_updates is not None and completed >= config.total_updates:
            raise ValueError("Checkpoint has already reached the requested update limit")
        selection = json.loads((output / "data_selection.json").read_text(encoding="utf-8"))
        for name, key in (("manifest", "manifest_sha256"), ("interval_index", "interval_index_sha256")):
            if hashlib.sha256(config.path(name).read_bytes()).hexdigest() != selection[key]:
                raise ValueError("Resume data manifest or interval index changed")
        history = restore_update_log(output / "metrics.jsonl", completed)
        window_history = restore_update_log(output / "training_windows.jsonl", completed)
        elapsed_before = max(history[-1]["elapsed_seconds"],
                             resume.stat().st_mtime-(output / "config.json").stat().st_mtime)
        shutil.copy2(output / "config.json", output / f"config.before_resume_{time.time_ns()}.json")
        write_json(output / "config.json", config.to_dict())
    actor, critic, normalizer, rollout = create_runtime(config, checkpoint)
    train_source = EpisodeSampler.from_config(config, actor.encoder, rollout, "train", preparation_pool=preparation_pool)
    if checkpoint is not None:
        if "data_sampler" in checkpoint:
            train_source.load_state_dict(checkpoint["data_sampler"])
        else:
            records = selection["train_initial_pool"] + [r for batch in window_history for r in batch["windows"]]
            rejected = selection["rejected_train"] + [r for batch in window_history for r in batch["rejected"]]
            train_source.restore_first_cycle(records, rejected)
        if (sorted(train_source.accepted) != checkpoint["pool_indices"]["train"]
                or train_source.summary() != history[-1]["training_data"]):
            raise ValueError("Recovered sampler coverage does not match the checkpointed update")
    initial_count = (config.train_windows if config.training_sampling == "fixed_pool" else
                     config.normalization_windows if config.emg_preprocessing == "standardize" else 0)
    train_episodes, train_indices, train_rejections = (train_source.draw(initial_count)
        if initial_count and checkpoint is None else ([], [], []))
    val_episodes, val_indices, val_rejections = select_episodes(config, actor, rollout, "val", config.validation_windows,
        selected_indices=None if checkpoint is None else checkpoint["pool_indices"]["val"], preparation_pool=preparation_pool)
    if config.emg_preprocessing == "standardize" and checkpoint is None:
        normalizer.fit([episode.sample["emg"] for episode in train_episodes[:config.normalization_windows]])
    if checkpoint is None:
        write_json(output / "data_selection.json", {
        "training_sampling": config.training_sampling,
        "training_jitter": config.training_jitter,
        "data_preparation_version": DATA_PREPARATION_VERSION,
        "train_initial_pool": train_source.records(train_indices, train_episodes),
        "training_window_log": "training_windows.jsonl",
        "val": [{"window_index": i, "filename": e.sample["filename"], "side": e.sample.get("side", "right"), "start": e.sample["start"],
                 "initialization": e.initialization}
                for i, e in zip(val_indices, val_episodes)],
        "rejected_train": train_rejections, "rejected_val": val_rejections,
        "emg_preprocessing": {"mode": config.emg_preprocessing,
            "fitted_training_windows": min(config.normalization_windows, len(train_episodes))
                if config.emg_preprocessing == "standardize" else 0,
            "input_clipping": config.normalization_clip if config.emg_preprocessing == "standardize" else None,
            "mean": normalizer.mean.cpu().tolist() if config.emg_preprocessing == "standardize" else None,
            "std": normalizer.std.cpu().tolist() if config.emg_preprocessing == "standardize" else None},
        "manifest_sha256": hashlib.sha256(config.path("manifest").read_bytes()).hexdigest(),
        "interval_index_sha256": hashlib.sha256(config.path("interval_index").read_bytes()).hexdigest(),
        "episode_semantics": "finite_reference_clip_terminal; reset_only_at_start; no_teacher_forcing",
        "control_period_seconds": rollout.period,
        "meta_encoder": actor.encoder.config.to_dict(), "online_causal": False,
        "initial_human_pose_source": "same_reference_time_as_robot_retarget",
        "reference_hand_convention": "canonical_right_from_unchanged_joint_angles",
        "torch": str(torch.__version__), "mujoco_model": str(rollout.environment.hand.source)})
    trainer = PPOTrainer(actor, critic, normalizer, rollout, config)
    pools = {"train": sorted(train_source.accepted), "val": val_indices}
    if checkpoint is None:
        initial_parameters = {name: p.detach().cpu().clone() for name, p in actor.named_parameters()}
        initial_eval, _ = evaluate_policy(actor, normalizer, rollout, val_episodes)
        hold_eval, _ = evaluate_policy(actor, normalizer, rollout, val_episodes, zero_actions=True)
        best_cost, best_update = initial_eval["mean_motion_cost"], 0
        save_checkpoint(output / "initial.pt", config, actor, critic, normalizer, rollout, trainer,
                        update=0, pools=pools, data_sampler=train_source)
        save_checkpoint(output / "best.pt", config, actor, critic, normalizer, rollout, trainer,
                        update=0, pools=pools, data_sampler=train_source)
        write_json(output / "initial_validation.json", {"untrained_mean_policy": initial_eval, "zero_action_hold": hold_eval})
        print(f"Initial validation cost={best_cost:.7f}; zero-action cost={hold_eval['mean_motion_cost']:.7f}", flush=True)
        history, evaluations = [], [{"update": 0, **initial_eval}]
    else:
        trainer.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        trainer.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        trainer.rng.bit_generator.state = checkpoint["trainer_rng"]
        initial_parameters = torch.load(output / "initial.pt", map_location="cpu", weights_only=True)["actor"]
        initial = json.loads((output / "initial_validation.json").read_text(encoding="utf-8"))
        initial_eval, hold_eval = initial["untrained_mean_policy"], initial["zero_action_hold"]
        evaluations = [{"update": 0, **initial_eval}] + [{"update": m["update"], "mean_motion_cost": m["validation_motion_cost"]}
                       for m in history if "validation_motion_cost" in m]
        best = min(evaluations, key=lambda m: m["mean_motion_cost"])
        best_cost, best_update = best["mean_motion_cost"], best["update"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        write_json(output / f"resume_{completed}.json", {"checkpoint_update": completed,
                   "restored_training_data": train_source.summary(), "new_config": config.to_dict()})
        print(f"Resumed update {completed}; restored {len(train_source.accepted)} training windows; best validation={best_cost:.7f}", flush=True)
    stopper = ValidationEarlyStopping(initial_eval["mean_motion_cost"], config)
    for evaluation in evaluations[1:]:
        stopper.observe(evaluation["update"], evaluation["mean_motion_cost"])
    stopped_early = False
    sampler = np.random.default_rng(config.seed + 202)
    mode = "x" if checkpoint is None else "a"
    with (output / "metrics.jsonl").open(mode, encoding="utf-8") as log, \
            (output / "training_windows.jsonl").open(mode, encoding="utf-8") as window_log:
        updates = (count(completed+1) if config.total_updates is None
                   else range(completed+1, config.total_updates+1))
        for update in updates:
            update_started = time.perf_counter()
            if config.training_sampling == "stream":
                episodes, indices, rejected = train_source.draw(config.episodes_per_update, repeat=True)
            elif config.training_jitter:
                episodes, indices, rejected = train_source.draw(config.episodes_per_update,
                                                               pool_indices=train_indices)
            else:
                selected = sampler.choice(len(train_episodes), config.episodes_per_update, replace=True)
                episodes = [train_episodes[i] for i in selected]
                indices, rejected = [train_indices[i] for i in selected], []
            window_log.write(json.dumps({"update": update, "windows": train_source.records(indices, episodes),
                                         "rejected": rejected}, allow_nan=False) + "\n")
            window_log.flush()
            preparation_seconds = time.perf_counter() - update_started
            collect_started = time.perf_counter()
            actions, old_log_prob, outcomes = trainer.collect(episodes)
            collect_seconds = time.perf_counter() - collect_started
            optimize_started = time.perf_counter()
            metrics = trainer.update(episodes, actions, old_log_prob, outcomes)
            metrics.update(preparation_seconds=preparation_seconds, collect_seconds=collect_seconds,
                           optimization_seconds=time.perf_counter()-optimize_started,
                           update_seconds=time.perf_counter()-update_started)
            metrics.update(update=update, environment_steps=update*config.episode_steps*config.episodes_per_update,
                           elapsed_seconds=elapsed_before+time.perf_counter()-started,
                           std_min=float(actor.motor_noise.stddev.detach().cpu()[rollout.mask].min()),
                           std_max=float(actor.motor_noise.stddev.detach().cpu()[rollout.mask].max()),
                           training_data=train_source.summary())
            if update % config.evaluate_every == 0 or update == config.total_updates:
                evaluation, _ = evaluate_policy(actor, normalizer, rollout, val_episodes)
                evaluations.append({"update": update, **evaluation})
                metrics["validation_motion_cost"] = evaluation["mean_motion_cost"]
                stopped_early = stopper.observe(update, evaluation["mean_motion_cost"])
                metrics["early_stopping"] = stopper.status()
                if evaluation["mean_motion_cost"] < best_cost:
                    best_cost, best_update = evaluation["mean_motion_cost"], update
                    pools["train"] = sorted(train_source.accepted)
                    save_checkpoint(output / "best.pt", config, actor, critic, normalizer, rollout, trainer,
                                    update=update, pools=pools, data_sampler=train_source)
            log.write(json.dumps(metrics, allow_nan=False) + "\n")
            log.flush()
            history.append(metrics)
            print(f"Update {update}/{config.total_updates or 'unlimited'}: cost={metrics['train_motion_cost']:.7f} "
                  f"KL={metrics['full_batch_analytic_kl']:.5f} steps={metrics['optimizer_steps']} "
                  f"actor_lr={metrics['actor_learning_rate']:.3g} "
                  f"clip={metrics['action_clipped_fraction']:.3f}" +
                  f" seconds={metrics['update_seconds']:.2f} prepare={preparation_seconds:.2f}" +
                  (f" val={metrics['validation_motion_cost']:.7f}" if 'validation_motion_cost' in metrics else ''), flush=True)
            if update % config.checkpoint_every == 0 or update == config.total_updates or stopped_early:
                pools["train"] = sorted(train_source.accepted)
                save_checkpoint(output / "last.pt", config, actor, critic, normalizer, rollout, trainer,
                                update=update, pools=pools, data_sampler=train_source)
            completed = update
            if stopped_early:
                print(f"Early stopping after {update} updates: {stopper.stale_checks} validation checks without meaningful improvement", flush=True)
                break
    changes = {}
    for component in ("encoder", "command_head", "synergy", "motor_noise"):
        changes[component] = float(sum((p.detach().cpu()-initial_parameters[name]).square().sum()
                                      for name, p in actor.named_parameters() if name.startswith(component+'.')).sqrt())
    summary = {"scope": "fixed_hand_ppo_experiment_not_converged_claim",
               "reward_objective": rollout.reward_settings,
               "updates": completed, "update_limit": config.total_updates,
               "stopped_early": stopped_early, "early_stopping": stopper.status(),
               "environment_steps": completed*config.episodes_per_update*config.episode_steps,
               "initial_validation": initial_eval, "zero_action_hold": hold_eval,
               "final_validation": evaluations[-1], "best_validation_cost": best_cost, "best_update": best_update,
               "parameter_change_l2": changes, "evaluations": evaluations,
               "training_data": train_source.summary(), "emg_preprocessing": config.emg_preprocessing,
               "episode_initialization": config.episode_initialization,
               "elapsed_seconds": elapsed_before+time.perf_counter()-started,
               "actor_inputs": ["EMG_history", "initial_human_pose",
                                "internally_accumulated_command_pose", "morphology"],
               "meta_frozen": True, "temporal_protocol": actor.encoder.config.temporal_protocol,
               "online_causal": False, "control_hz": 50, "test_split_used": False,
               "simulation_warnings": sum(w.number for w in rollout.environment.data.warning)}
    write_json(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("environment_steps", "best_validation_cost", "best_update", "parameter_change_l2", "elapsed_seconds")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_DIR / "experiments/rl_meta_pose_feedback/configs/ppo.json")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this training run")
    parser.add_argument("--resume", type=Path, help="Resume this output directory's last.pt (stream mode)")
    parser.add_argument("--mjcf", type=Path, help="Override the candidate MJCF")
    parser.add_argument("--updates", type=int, help="Override this trial's update budget")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    config = TrainingConfig.load(args.config)
    overrides = {name: value for name, value in {"total_updates": args.updates, "device": args.device,
                 "mjcf": None if args.mjcf is None else str(args.mjcf.resolve())}.items() if value is not None}
    run_training(replace(config, **overrides), args.output, resume=args.resume)


if __name__ == "__main__":
    main()
