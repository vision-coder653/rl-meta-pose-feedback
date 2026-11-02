"""Parallel initialization must preserve sampling, rejection and resume semantics."""
from pathlib import Path
import sys
APP = Path(__file__).resolve().parents[3]
for source in (APP, APP / "core", APP / "experiments"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from types import SimpleNamespace
import unittest
import numpy as np
import torch

from rl_meta_pose_feedback.training.config import TrainingConfig
from rl_meta_pose_feedback.training.runtime import create_runtime
from rl_meta_pose_feedback.training.parallel import RetargetPool, _solve
from rl_meta_pose_feedback.training.sampling import EpisodeSampler
from rl_meta_pose_feedback.training.rollout import ReferenceCoverageError, REFERENCE_COVERAGE_ERROR


class Dataset:
    def __len__(self):
        return 7

    def __getitem__(self, index):
        return dict(index=index, filename=str(index), start=index*100, stop=index*100+90,
                    segment_id=0, jitter_samples=0)

    def sample(self, index, *, rng):
        result = self[index]
        offset = int(rng.integers(10))
        result.update(start=result["start"]+offset, stop=result["stop"]+offset, jitter_samples=offset)
        return result


class ParallelPreparationTests(unittest.TestCase):
    def sampler(self, parallel, jitter):
        def prepare(sample):
            if sample["index"] in (1, 4):
                raise ReferenceCoverageError(REFERENCE_COVERAGE_ERROR)
            return SimpleNamespace(sample=sample)
        def many(samples):
            results = []
            for sample in samples:
                try:
                    results.append(prepare(sample))
                except ReferenceCoverageError as error:
                    results.append(error)
            return results
        return EpisodeSampler(Dataset(), prepare, gap_by_name={str(i): .001 for i in range(7)},
            user_by_name={str(i): str(i) for i in range(7)}, seed=3, jitter=jitter,
            prepare_many=many if parallel else None, preparation_batch_size=4)

    def test_sampling_rejections_cycles_and_resume_match_serial(self):
        for jitter in (False, True):
            serial, parallel = self.sampler(False, jitter), self.sampler(True, jitter)
            for count in (3, 8, 6):
                a, ai, ar = serial.draw(count, repeat=True)
                b, bi, br = parallel.draw(count, repeat=True)
                self.assertEqual(serial.records(ai, a), parallel.records(bi, b))
                self.assertEqual(ar, br)
                self.assertEqual(serial.state_dict(), parallel.state_dict())
                restored = self.sampler(True, jitter)
                restored.load_state_dict(serial.state_dict())
                parallel = restored

    def test_fixed_and_pool_selection_match_and_fail_closed(self):
        for mode in ({"selected_indices": [6, 2, 0]}, {"pool_indices": [0, 1, 2, 4, 6]}):
            serial, parallel = self.sampler(False, True), self.sampler(True, True)
            a, ai, ar = serial.draw(3, **mode)
            b, bi, br = parallel.draw(3, **mode)
            self.assertEqual(serial.records(ai, a), parallel.records(bi, b))
            self.assertEqual(ar, br)
            self.assertEqual(serial.state_dict(), parallel.state_dict())
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            self.sampler(True, False).draw(2, selected_indices=[0, 1])

    def test_spawned_solver_matches_serial_and_shuts_down(self):
        config = TrainingConfig(device="cpu", episode_steps=8, retarget_workers=2)
        _, _, _, rollout = create_runtime(config)
        count = 1790 + 40*9 + 1
        samples = [dict(emg=torch.zeros(count, 16), time=torch.arange(count, dtype=torch.float64)/2000,
                        joint_angles=torch.full((count, 20), angle), max_gap_seconds=.00075,
                        filename=str(angle), start=0, stop=count) for angle in (0., .2)]
        bad = {**samples[0], "time": samples[0]["time"] / 2}
        # The parent's preloaded BLAS may use more threads than the workers.
        # Tiny solver-rounding differences are allowed, not geometry changes.
        expected = [rollout.prepare(s, left_context=1790, stride=40) for s in samples]
        pool = RetargetPool(config)
        with pool:
            actual = pool.prepare_many([*samples, bad], rollout, left_context=1790, stride=40)
            for a, b in zip(expected, actual):
                np.testing.assert_allclose(a.initial_angles, b.initial_angles, rtol=0, atol=1e-5)
                torch.testing.assert_close(a.human_points, b.human_points, rtol=0, atol=0)
                self.assertEqual(a.initialization["retargeting"]["status"], b.initialization["retargeting"]["status"])
                self.assertAlmostEqual(a.initialization["retargeting"]["cost"], b.initialization["retargeting"]["cost"], places=10)
            self.assertIsInstance(actual[-1], ReferenceCoverageError)
            processes = list(pool.executor._processes.values())
            with self.assertRaises(ValueError):
                pool.executor.submit(_solve, np.zeros(3, dtype=np.float32)).result()
        self.assertTrue(all(not process.is_alive() for process in processes))
        self.assertIsNone(pool.executor)

    def test_worker_setting_validation(self):
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                TrainingConfig(retarget_workers=value)


if __name__ == "__main__":
    unittest.main()
