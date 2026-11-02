"""Persistent CPU-only retarget workers; samples and RNG stay in the parent."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os

import torch

from configs.groups import load_training_groups
from kinematics import RobotLandmarks
from simulation.morphology import HandMorphology
from simulation.retargeting import PoseRetargeter
from rl_meta_pose_feedback.training.rollout import ReferenceCoverageError, RetargetCollisionError

_solver = None
_initial = None


def _initialize_worker(config):
    global _solver, _initial
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    hand = HandMorphology.load(config.path("mjcf"))
    group = config.reward_group(load_training_groups(config.path("group_config"))[config.group])
    _solver = PoseRetargeter(hand, RobotLandmarks.from_hod(hand), group, config.retargeting)
    _initial = hand.validate_angles(config.initial_angles)


def _solve(angles):
    # Only a 20-angle NumPy array crosses into each worker, never CUDA tensors.
    return _solver.solve(torch.from_numpy(angles), initial_angles=_initial)


class RetargetPool:
    def __init__(self, config):
        self.config = config
        self.workers = config.retarget_workers if config.episode_initialization == "retarget" else 0
        self.executor = None
        self._environment = {}

    def __enter__(self):
        if self.workers:
            # Spawned native libraries see these before importing NumPy/Torch.
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                self._environment[key] = os.environ.get(key)
                os.environ[key] = "1"
            self.executor = ProcessPoolExecutor(max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker, initargs=(self.config,))
        return self

    def __exit__(self, *exc):
        try:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
        finally:
            self.executor = None
            for key, value in self._environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def prepare_many(self, samples, rollout, *, left_context, stride):
        pending = []
        for sample in samples:
            try:
                episode = rollout.prepare(sample, left_context=left_context, stride=stride, initialize=False)
            except ReferenceCoverageError as error:
                pending.append((error, None))
                continue
            future = self.executor.submit(_solve, episode.human_initial_pose.numpy())
            pending.append((episode, future))
        # Consume in input order, irrespective of completion order. Unexpected
        # worker errors propagate; only the original two rejection types skip.
        results = []
        for episode, future in pending:
            if future is None:
                results.append(episode)
                continue
            result = future.result()
            try:
                results.append(rollout.initialize_episode(episode, result))
            except RetargetCollisionError as error:
                results.append(error)
        return results
