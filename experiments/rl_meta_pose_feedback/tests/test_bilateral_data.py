"""Hand sides share user splits; EMG/angles pass through without mirroring."""
from pathlib import Path
import sys
APP = Path(__file__).resolve().parents[3]
for source in (APP, APP / "core", APP / "experiments"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import csv
import json
import tempfile
import unittest
import h5py
import numpy as np
import torch
from emg_control_paths import CACHE_DIR
from rl_meta_pose_feedback.training.data import make_split, load_split_manifest, index_recordings, load_indexed_split
from rl_meta_pose_feedback.training.windows import TrackingWindowDataset
from data_pipeline.windows import make_right_hand_split
from kinematics.human import UmeTrackFK


class BilateralDataTests(unittest.TestCase):
    def setUp(self):
        folder = CACHE_DIR / "bilateral_tests"
        folder.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=folder)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = []
        dtype = np.dtype([("time", "f8"), ("emg", "f4", (16,)), ("joint_angles", "f4", (20,))])
        self.raw = {}
        for user in range(4):
            for side in ("left", "right"):
                name = f"u{user}_{side}.hdf5"
                row = dict(filename=name, user=f"u{user}", session=f"session{user}", side=side,
                           stage="pinch", split="train", moving_hand=side)
                rows.append(row)
                values = np.zeros(200, dtype=dtype)
                values["time"] = np.arange(200) / 2000.
                values["emg"] = np.arange(16) - 7 + (user if side == "left" else -user)
                values["joint_angles"] = np.linspace(-.2, .3, 20)
                # Both sides have a real gap. Windows must never bridge it.
                values["time"][100:] += .01
                self.raw[name] = values
                with h5py.File(self.root / name, "w") as f:
                    f.create_dataset("emg2pose/timeseries", data=values)
        with (self.root / "metadata.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_shared_user_split_and_right_only_compatibility(self):
        both = make_split(self.root)
        original = make_right_hand_split(self.root)
        old_users = {r["metadata"]["user"]: r["split"] for r in original["records"]}
        self.assertEqual(len(both["records"]), 8)
        for record in both["records"]:
            self.assertEqual(record["split"], old_users[record["metadata"]["user"]])
        path = self.root / "split.json"
        path.write_text(json.dumps(original))
        self.assertEqual(load_split_manifest(path), original)
        both["records"][1]["split"] = "val" if both["records"][0]["split"] != "val" else "train"
        path.write_text(json.dumps(both))
        with self.assertRaisesRegex(ValueError, "multiple"):
            load_split_manifest(path)

    def test_both_sides_preserve_arrays_and_segment_boundaries(self):
        manifest = make_split(self.root)
        path, index_path = self.root / "split.json", self.root / "index.jsonl"
        path.write_text(json.dumps(manifest))
        index_recordings(manifest, index_path, data_dir=self.root, workers=1)
        indexes = load_indexed_split(path, index_path, "train", data_dir=self.root)
        dataset = TrackingWindowDataset(indexes, window_samples=50, window_stride=50,
                                        first_output_sample=10, output_stride=10)
        sides = set()
        for i in range(len(dataset)):
            sample = dataset.sample(i, rng=np.random.default_rng(i))
            sides.add(sample["side"])
            expected = self.raw[sample["filename"]][sample["start"]:sample["stop"]]
            torch.testing.assert_close(sample["emg"], torch.from_numpy(expected["emg"].copy()))
            torch.testing.assert_close(sample["joint_angles"], torch.from_numpy(expected["joint_angles"].copy()))
            self.assertFalse(sample["start"] < 100 < sample["stop"])
        self.assertEqual(sides, {"left", "right"})
        # A complete index is required, even if omitted rows belong to another split.
        lines = index_path.read_text().splitlines()
        index_path.write_text("\n".join(lines[:-1])+"\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_indexed_split(path, index_path, "train", data_dir=self.root)

    def test_reuse_is_validated_and_wrong_side_metadata_rejected(self):
        manifest = make_split(self.root)
        path = self.root / "split.json"
        path.write_text(json.dumps(manifest))
        old, new = self.root / "old.jsonl", self.root / "new.jsonl"
        index_recordings(manifest, old, data_dir=self.root, workers=1)
        summary = index_recordings(manifest, new, data_dir=self.root, workers=1, reuse_index=old)
        self.assertEqual(summary["statistics"]["reused_indexes"], 8)
        rows = [json.loads(line) for line in new.read_text().splitlines()]
        rows[0]["metadata"]["side"] = "right" if rows[0]["metadata"]["side"] == "left" else "left"
        new.write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_indexed_split(path, new, "train", data_dir=self.root)

    def test_same_angles_produce_mirrored_geometry_and_proper_frames(self):
        q = torch.linspace(-.2, .5, 60, dtype=torch.float64).reshape(3, 20)
        left, right = UmeTrackFK(side="left"), UmeTrackFK(side="right")
        mirror = torch.diag(torch.tensor([-1., 1., 1.], dtype=torch.float64))
        axes = torch.diag(torch.tensor([1., -1., 1.], dtype=torch.float64))
        torch.testing.assert_close(right(q), left(q) @ mirror.T)
        torch.testing.assert_close(right.fingertip_frames(q), mirror @ left.fingertip_frames(q) @ axes)
        torch.testing.assert_close(torch.linalg.det(right.fingertip_frames(q)), torch.ones(3, 5, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
