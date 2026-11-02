"""Right-hand, user-disjoint splits and windows inside indexed valid segments.

The recordings module owns signal validity rules. Paths remain anchored
to the experiment root, independently of the working directory.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from data_pipeline.recordings import Recording, RecordingIndex, iter_indexes, load_recordings, read_window, scan_recording


from emg_control_paths import DATA_DIR, EXPERIMENT_DIR, SHARED_DATA_DIR, resolve_experiment_path

DEFAULT_DATA_DIR = DATA_DIR
SPLITS = ("train", "val", "test")


def make_right_hand_split(data_dir: Path = DEFAULT_DATA_DIR, *, seed: int = 42) -> dict:
    """Split right-hand recordings by user, approximately 80/10/10 users.

    All stages and moving_hand values are retained; technical validity is a
    separate indexing step. Original official split metadata is preserved.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    recordings = [r for r in load_recordings(data_dir) if r.metadata["side"] == "right"]
    users = sorted({r.metadata["user"] for r in recordings})
    if len(users) < 3:
        raise ValueError("Need at least three right-hand users for three disjoint splits")
    # Hash ordering is stable across Python versions and metadata row order.
    users.sort(key=lambda user: hashlib.sha256(f"{seed}:{user}".encode()).digest())
    held_out = max(1, len(users) // 10)
    assignments = {user: "test" if i < held_out else "val" if i < 2 * held_out else "train"
                   for i, user in enumerate(users)}
    records = [{"filename": r.path.name, "split": assignments[r.metadata["user"]],
                "metadata": r.metadata} for r in sorted(recordings, key=lambda r: r.path.name)]
    return {
        "schema_version": 1, "side": "right", "group_by": "user", "seed": seed,
        "requested_user_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "metadata_sha256": hashlib.sha256((Path(data_dir) / "metadata.csv").read_bytes()).hexdigest(),
        "summary": {split: {
            "users": sum(value == split for value in assignments.values()),
            "recordings": sum(r["split"] == split for r in records),
        } for split in SPLITS},
        "records": records,
    }


def load_split_manifest(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("side") != "right"
            or manifest.get("group_by") != "user"):
        raise ValueError("Unsupported right-hand split manifest")
    seen, user_split = set(), {}
    for record in manifest["records"]:
        name, split, metadata = record["filename"], record["split"], record["metadata"]
        if (Path(name).name != name or "/" in name or "\\" in name or name in seen
                or split not in SPLITS or metadata.get("side") != "right"):
            raise ValueError("Manifest has invalid, duplicate, or non-right-hand recordings")
        seen.add(name)
        user = metadata["user"]
        if user in user_split and user_split[user] != split:
            raise ValueError("One user appears in multiple experiment splits")
        user_split[user] = split
    if not seen or set(user_split.values()) != set(SPLITS):
        raise ValueError("Manifest must contain all three nonempty splits")
    return manifest


def index_right_recordings(manifest: dict, output: Path, *, data_dir: Path = DEFAULT_DATA_DIR,
                           workers: int = 4, max_gap_seconds: float = 0.00075) -> dict:
    """Index every manifest recording once; report failures instead of dropping them."""
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive and finite")
    output = Path(output)
    if output.suffix != ".jsonl":
        raise ValueError("Interval index must use .jsonl")
    records = manifest["records"]

    def scan(record: dict) -> dict:
        metadata = dict(record["metadata"], experiment_split=record["split"])
        try:
            index = scan_recording(Recording(Path(data_dir) / record["filename"], metadata),
                                   max_gap_seconds=max_gap_seconds)
            return dict(index.to_dict(), status="ok")
        except (OSError, KeyError, ValueError) as error:
            return {"filename": record["filename"], "status": "error", "error": str(error)}

    output.parent.mkdir(parents=True, exist_ok=True)
    totals = Counter({"recordings": 0, "errors": 0})
    with output.open("x", encoding="utf-8") as file, ThreadPoolExecutor(workers) as pool:
        for number, item in enumerate(pool.map(scan, records), 1):
            file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            if item["status"] == "ok":
                totals["recordings"] += 1
                for key in ("samples", "valid_samples", "invalid_samples", "segments",
                            "valid_duration_seconds", "gap_edges", "nonincreasing_time_edges"):
                    totals[key] += item["statistics"][key]
            else:
                totals["errors"] += 1
            if number == 1 or number % 100 == 0 or number == len(records):
                file.flush()
                print(f"Right-hand index: {number}/{len(records)}, errors={totals['errors']}", flush=True)
    summary = {"scope": "all_right_hand_recordings", "split_by": "user",
               "manifest_recordings": len(records), "max_gap_seconds": max_gap_seconds,
               "statistics": dict(totals)}
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    if totals["errors"]:
        raise RuntimeError(f"{totals['errors']} recordings failed indexing; inspect {output}")
    return summary


def load_indexed_split(manifest_path: Path, index_path: Path, split: str, *,
                       data_dir: Path = DEFAULT_DATA_DIR) -> list[RecordingIndex]:
    """Load a complete index and select the requested experiment split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    manifest = load_split_manifest(manifest_path)
    expected = {r["filename"]: r for r in manifest["records"]}
    seen, selected = set(), []
    for index in iter_indexes(index_path, data_dir=data_dir):
        name = index.recording.path.name
        if name not in expected or name in seen:
            raise ValueError("Interval index contains an unexpected or duplicate recording")
        record = expected[name]
        metadata = index.recording.metadata
        if (metadata.get("side") != "right" or metadata.get("user") != record["metadata"]["user"]
                or metadata.get("experiment_split") != record["split"]):
            raise ValueError("Interval index does not match the split manifest")
        seen.add(name)
        if record["split"] == split:
            selected.append(index)
    if seen != set(expected):
        raise ValueError("Interval index is incomplete; refusing a partial training split")
    return selected


class EMGWindowDataset(Dataset):
    """Fixed-size windows, never crossing an invalid sample or timestamp break.

    Window size, window stride, and encoder alignment are explicit caller
    choices. Opening the HDF5 in read_window is safe for Windows DataLoader
    workers. This class holds no open HDF5 handles or raw signal arrays.
    """

    def __init__(self, indexes: Sequence[RecordingIndex], *, window_samples: int,
                 window_stride: int, first_output_sample: int, output_stride: int):
        for name, value in (("window_samples", window_samples), ("window_stride", window_stride),
                            ("output_stride", output_stride)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(first_output_sample) is not int or not 0 <= first_output_sample < window_samples:
            raise ValueError("first_output_sample must be inside the window")
        self.indexes = tuple(indexes)
        self.window_samples, self.window_stride = window_samples, window_stride
        self.label_indices = torch.arange(first_output_sample, window_samples, output_stride)
        self._segments, self._ends = [], []
        total = 0
        for recording_id, index in enumerate(self.indexes):
            if index.recording.metadata.get("side") != "right":
                raise ValueError("EMGWindowDataset currently accepts right-hand recordings only")
            for segment_id, segment in enumerate(index.segments):
                count = max(0, (segment.samples - window_samples) // window_stride + 1)
                if count:
                    self._segments.append((recording_id, segment_id))
                    total += count
                    self._ends.append(total)

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, item: int) -> dict:
        if not 0 <= item < len(self):
            raise IndexError("window index out of range")
        block = bisect.bisect_right(self._ends, item)
        local = item - (self._ends[block - 1] if block else 0)
        recording_id, segment_id = self._segments[block]
        window = read_window(self.indexes[recording_id], segment_id,
                             offset=local * self.window_stride, length=self.window_samples)
        emg = torch.from_numpy(window["emg"]).float()
        angles = torch.from_numpy(window["joint_angles"]).float()
        time = torch.from_numpy(window["time"]).double()
        return {
            "emg": emg, "joint_angles": angles, "time": time,
            "target_angles": angles[self.label_indices],
            "target_time": time[self.label_indices], "label_indices": self.label_indices.clone(),
            "filename": window["source_filename"], "segment_id": segment_id,
            "start": window["start"], "stop": window["stop"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path,
                        default=SHARED_DATA_DIR / "right_split.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", type=Path, help="Optional full right-hand .jsonl interval index")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.manifest = resolve_experiment_path(args.manifest)
    if args.index is not None:
        args.index = resolve_experiment_path(args.index)
    manifest = make_right_hand_split(args.data_dir, seed=args.seed)
    if args.manifest.exists():
        if load_split_manifest(args.manifest) != manifest:
            parser.error("Existing manifest differs; choose a new output path")
    else:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("x", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(manifest["summary"], indent=2), flush=True)
    if args.index:
        index_right_recordings(manifest, args.index, data_dir=args.data_dir, workers=args.workers)


if __name__ == "__main__":
    main()
