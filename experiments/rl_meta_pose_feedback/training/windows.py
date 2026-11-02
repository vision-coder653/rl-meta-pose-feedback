"""Experiment-owned window jitter; shared code only owns recording validity/indexes."""
import bisect
import numpy as np
import torch

from torch.utils.data import Dataset
from data_pipeline.recordings import read_window


class TrackingWindowDataset(Dataset):
    """Keep this experiment's window extraction independent of shared sampler edits."""

    def __init__(self, indexes, *, window_samples: int,
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
            if index.recording.metadata.get("side") not in ("left", "right"):
                raise ValueError("Window index must declare left or right hand")
            for segment_id, segment in enumerate(index.segments):
                count = max(0, (segment.samples - window_samples) // window_stride + 1)
                if count:
                    self._segments.append((recording_id, segment_id))
                    total += count
                    self._ends.append(total)

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, item):
        return self.sample(item)

    def sample(self, item, *, rng: np.random.Generator | None = None):
        if not 0 <= item < len(self):
            raise IndexError("window index out of range")
        block = bisect.bisect_right(self._ends, item)
        local = item - (self._ends[block - 1] if block else 0)
        recording_id, segment_id = self._segments[block]
        offset = local * self.window_stride
        segment = self.indexes[recording_id].segments[segment_id]
        limit = min(self.window_stride, segment.samples - offset - self.window_samples)
        jitter = int(rng.integers(limit)) if rng is not None and limit > 0 else 0
        window = read_window(self.indexes[recording_id], segment_id,
                             offset=offset + jitter, length=self.window_samples)
        emg = torch.from_numpy(window["emg"]).float()
        angles = torch.from_numpy(window["joint_angles"]).float()
        time = torch.from_numpy(window["time"]).double()
        return {
            "emg": emg, "joint_angles": angles, "time": time,
            "target_angles": angles[self.label_indices],
            "target_time": time[self.label_indices], "label_indices": self.label_indices.clone(),
            "filename": window["source_filename"], "side": window["metadata"]["side"], "segment_id": segment_id,
            "start": window["start"], "stop": window["stop"], "jitter_samples": jitter,
        }
