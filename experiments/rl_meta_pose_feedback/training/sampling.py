"""Lazy, reproducible sampling across a complete user-disjoint data split."""
from __future__ import annotations

from collections import Counter

import numpy as np

from rl_meta_pose_feedback.training.data import load_indexed_split
from rl_meta_pose_feedback.training.windows import TrackingWindowDataset
from rl_meta_pose_feedback.training.rollout import (REFERENCE_COVERAGE_ERROR, ReferenceCoverageError,
                     RETARGET_COLLISION_ERROR, RetargetCollisionError)


# Kept only to recognize historical sampler logs, not to reject new windows.
CLOCK_MISMATCH = "EMG output timestamps drift beyond the configured physics tolerance"


class EpisodeSampler:
    """Read one window at a time; exhaust shuffled candidates before repeating.

    Insufficient reference coverage and collision-rejected retarget poses are
    skipped after raw validity indexing. Clock drift is not a quality filter. Corrupt files, changed
    sources and unexpected preparation failures still stop the run.
    Raw windows and prepared FK arrays are owned by the caller, not cached here.
    """

    def __init__(self, dataset, prepare, *, gap_by_name, user_by_name, seed, jitter=False,
                 prepare_many=None, preparation_batch_size=1):
        if not len(dataset):
            raise ValueError("No window fits the requested episode and EMG history")
        self.dataset, self.prepare = dataset, prepare
        self.prepare_many, self.preparation_batch_size = prepare_many, preparation_batch_size
        self.jitter = jitter
        self.gap_by_name, self.user_by_name = gap_by_name, user_by_name
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(dataset))
        self.cursor, self.cycles, self.attempts, self.draws = 0, 0, 0, 0
        self.rejected, self.accepted = set(), set()
        self.recordings, self.users = set(), set()
        self.reasons = Counter()

    @classmethod
    def from_config(cls, config, encoder, rollout, split, *, initialize=True, preparation_pool=None):
        indexes = load_indexed_split(config.path("manifest"), config.path("interval_index"), split,
                                     data_dir=config.path("data_dir"))
        stride, context = encoder.control_sample_stride, encoder.frontend_left_context
        # Extra raw samples cover the step-end reference and a guard interval.
        size = context + 1 + stride * (config.episode_steps + 1)
        dataset = TrackingWindowDataset(indexes, window_samples=size, window_stride=size,
                                  first_output_sample=context, output_stride=stride)
        return cls(dataset, lambda sample: rollout.prepare(sample, left_context=context, stride=stride,
                                                           initialize=initialize),
                   gap_by_name={i.recording.path.name: i.max_gap_seconds for i in indexes},
                   user_by_name={i.recording.path.name: i.recording.metadata["user"] for i in indexes},
                   seed=config.seed + {"train": 0, "val": 1, "test": 2}[split],
                   jitter=split == "train" and config.training_jitter,
                   prepare_many=(lambda samples: preparation_pool.prepare_many(samples, rollout,
                       left_context=context, stride=stride))
                       if initialize and preparation_pool is not None and preparation_pool.workers else None,
                   preparation_batch_size=preparation_pool.workers if preparation_pool is not None else 1)

    def draw(self, count, *, repeat=False, selected_indices=None, pool_indices=None):
        if type(count) is not int or count <= 0:
            raise ValueError("Requested episode count must be positive")
        fixed = None if selected_indices is None else list(selected_indices)
        pool = None if pool_indices is None else list(pool_indices)
        if pool is not None and (fixed is not None or not pool or len(set(pool)) != len(pool)
                or any(type(i) is not int or not 0 <= i < len(self.dataset) for i in pool)):
            raise ValueError("Training pool must contain distinct valid window indices")
        if fixed is not None and (len(fixed) != count or len(set(fixed)) != count
                or any(type(i) is not int or not 0 <= i < len(self.dataset) for i in fixed)):
            raise ValueError("Frozen selection must contain the requested distinct window indices")
        episodes, accepted, rejected = [], [], []
        while len(episodes) < count:
            batch = []
            # Pool sampling depends on each rejection removing a candidate;
            # retain serial selection for that legacy mode. Stream/fixed
            # selections batch only the remaining demand, without lookahead.
            capacity = min(count - len(episodes), self.preparation_batch_size or 1) if self.prepare_many and pool is None else 1
            while len(batch) < capacity:
                if pool is not None:
                    if not pool:
                        raise ValueError("Insufficient compatible windows in the training pool")
                    index = int(self.rng.choice(pool))
                elif fixed is not None:
                    index = fixed[len(episodes) + len(batch)]
                else:
                    if self.cursor == len(self.order):
                        if batch:
                            break  # Commit this cycle before drawing the next permutation.
                        if not repeat or not self.accepted:
                            raise ValueError(f"Insufficient compatible windows: requested={count}, accepted={len(episodes)}")
                        self.order = self.rng.permutation(len(self.dataset))
                        self.cursor = 0
                        self.cycles += 1
                    index = int(self.order[self.cursor])
                    self.cursor += 1
                    # A failed jittered variant does not invalidate its whole grid cell.
                    if index in self.rejected and not self.jitter:
                        continue
                sample = (self.dataset.sample(index, rng=self.rng) if self.jitter and fixed is None
                          else self.dataset[index])
                sample["max_gap_seconds"] = self.gap_by_name[sample["filename"]]
                batch.append((index, sample))
            if self.prepare_many is not None:
                prepared_batch = self.prepare_many([sample for _, sample in batch])
            else:
                try:
                    prepared_batch = [self.prepare(batch[0][1])]
                except (ReferenceCoverageError, RetargetCollisionError) as error:
                    prepared_batch = [error]
            if len(prepared_batch) != len(batch):
                raise RuntimeError("Preparation batch length differs from selected samples")
            for (index, sample), prepared in zip(batch, prepared_batch):
                self.attempts += 1
                try:
                    episode = prepared
                    if isinstance(episode, Exception):
                        raise episode
                except (ReferenceCoverageError, RetargetCollisionError) as error:
                    if fixed is not None:
                        raise ValueError("Previously selected evaluation window no longer matches") from error
                    self.rejected.add(index)
                    reason = RETARGET_COLLISION_ERROR if isinstance(error, RetargetCollisionError) else str(error)
                    self.reasons[reason] += 1
                    rejected.append({"window_index": index, "reason": reason})
                    if isinstance(error, RetargetCollisionError):
                        rejected[-1].update(error.details)
                    if self.jitter:
                        rejected[-1].update(start=sample["start"], stop=sample["stop"],
                                            jitter_samples=sample["jitter_samples"])
                    if pool is not None:
                        pool.remove(index)
                    continue
                episodes.append(episode)
                accepted.append(index)
                self.accepted.add(index)
                self.recordings.add(sample["filename"])
                self.users.add(self.user_by_name[sample["filename"]])
                self.draws += 1
        return episodes, accepted, rejected

    def records(self, indices, episodes):
        return [{"window_index": i, "filename": e.sample["filename"],
                 "user": self.user_by_name[e.sample["filename"]], "side": e.sample.get("side", "right"),
                 "segment_id": e.sample["segment_id"], "start": e.sample["start"], "stop": e.sample["stop"],
                 "initialization": getattr(e, "initialization", None),
                 "jitter_samples": e.sample.get("jitter_samples", 0)}
                for i, e in zip(indices, episodes)]

    def summary(self):
        return {"candidate_windows": len(self.dataset), "read_attempts": self.attempts,
                "accepted_draws": self.draws, "unique_windows": len(self.accepted),
                "recordings": len(self.recordings), "users": len(self.users),
                "completed_candidate_cycles": self.cycles,
                "rejected_windows": len(self.rejected),
                "reference_coverage_rejected_windows": self.reasons[REFERENCE_COVERAGE_ERROR],
                "clock_rejected_windows": self.reasons[CLOCK_MISMATCH],
                "retarget_collision_rejected_draws": self.reasons[RETARGET_COLLISION_ERROR],
                "rejection_reasons": dict(self.reasons)}

    def state_dict(self):
        return {"order": self.order.tolist(), "rng": self.rng.bit_generator.state, "jitter": self.jitter,
                "cursor": self.cursor, "cycles": self.cycles, "attempts": self.attempts, "draws": self.draws,
                "accepted": sorted(self.accepted), "rejected": sorted(self.rejected),
                "recordings": sorted(self.recordings), "users": sorted(self.users), "reasons": dict(self.reasons)}

    def load_state_dict(self, state):
        order = np.asarray(state["order"], dtype=np.int64)
        if (state.get("jitter", False) != self.jitter
                or order.shape != (len(self.dataset),) or not np.array_equal(np.sort(order), np.arange(len(order)))
                or not 0 <= state["cursor"] <= len(order)):
            raise ValueError("Sampler state does not match the current dataset")
        self.order = order
        self.rng.bit_generator.state = state["rng"]
        self.cursor, self.cycles = state["cursor"], state["cycles"]
        self.attempts, self.draws = state["attempts"], state["draws"]
        self.accepted, self.rejected = set(state["accepted"]), set(state["rejected"])
        self.recordings, self.users = set(state["recordings"]), set(state["users"])
        self.reasons = Counter(state["reasons"])

    def restore_first_cycle(self, records, rejections):
        """Recover older stream checkpoints from their complete per-update log.

        A permutation prefix is verified, so missing or reordered data cannot
        silently resume at a different point. New checkpoints store full state.
        """
        if self.jitter:
            raise ValueError("Cannot reconstruct jitter RNG from old window logs; sampler state is required")
        accepted = {r["window_index"] for r in records}
        rejected = {r["window_index"] for r in rejections}
        visited = accepted | rejected
        if (len(accepted) != len(records) or len(rejected) != len(rejections) or accepted & rejected
                or visited != set(self.order[:len(visited)].tolist())
                or any(r["reason"] not in (CLOCK_MISMATCH, REFERENCE_COVERAGE_ERROR, RETARGET_COLLISION_ERROR) for r in rejections)
                or any(self.user_by_name.get(r["filename"]) != r["user"] for r in records)):
            raise ValueError("Cannot reconstruct an exact sampler prefix from the old checkpoint logs")
        self.cursor = self.attempts = len(visited)
        self.draws = len(records)
        self.accepted, self.rejected = accepted, rejected
        self.recordings = {r["filename"] for r in records}
        self.users = {r["user"] for r in records}
        self.reasons = Counter(r["reason"] for r in rejections)
