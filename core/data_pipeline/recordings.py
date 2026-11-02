"""Read emg2pose and index continuous valid intervals without changing signals.

Reference: Meta emg2pose, commit 5f6f62b1a0a08426adffe55900842e75a8adb38c,
emg2pose/data.py and utils.py. The gap check is an additional project rule,
not a threshold provided by Meta. Intervals consistently use [start, stop).
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


from emg_control_paths import DATA_DIR

DEFAULT_DATA_DIR = DATA_DIR
DEFAULT_MAX_GAP_SECONDS = 0.00075


@dataclass(frozen=True)
class Recording:
    path: Path
    metadata: dict[str, str]


@dataclass(frozen=True)
class Segment:
    start: int
    stop: int
    start_time: float
    end_time: float
    start_reason: str

    @property
    def samples(self) -> int:
        return self.stop - self.start

    @property
    def duration_seconds(self) -> float:
        # Elapsed time between observed endpoints; one sample has duration zero.
        return self.end_time - self.start_time


@dataclass(frozen=True)
class RecordingIndex:
    recording: Recording
    max_gap_seconds: float
    segments: tuple[Segment, ...]
    statistics: dict[str, int | float]
    source_size: int
    source_mtime_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "filename": self.recording.path.name,
            "metadata": self.recording.metadata,
            "max_gap_seconds": self.max_gap_seconds,
            "statistics": self.statistics,
            "source_size": self.source_size,
            "source_mtime_ns": self.source_mtime_ns,
            "segments": [asdict(segment) for segment in self.segments],
        }


def load_recordings(data_dir: Path = DEFAULT_DATA_DIR) -> list[Recording]:
    """Load every metadata row, preserving official splits and all hand sides.

    Selection for an experiment is deliberately separate from technical validity.
    This function opens no HDF5 handles, including when used by worker processes.
    """
    data_dir = Path(data_dir)
    required = {"filename", "session", "user", "side", "stage", "split", "moving_hand"}
    recordings = []
    seen = set()
    with (data_dir / "metadata.csv").open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"metadata.csv missing columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            if any(not row.get(key) for key in required):
                raise ValueError(f"metadata.csv line {line}: missing required value")
            name = row["filename"]
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError(f"metadata.csv line {line}: filename must be a basename")
            name = name if name.endswith(".hdf5") else name + ".hdf5"
            if name in seen:
                raise ValueError(f"Duplicate recording filename: {name}")
            seen.add(name)
            if row["side"] not in {"left", "right"} or row["split"] not in {"train", "val", "test"}:
                raise ValueError(f"metadata.csv line {line}: unexpected side or split")
            recordings.append(Recording(data_dir / name, dict(row)))
    return recordings


def _validate_schema(dataset: h5py.Dataset) -> None:
    if dataset.ndim != 1:
        raise ValueError("emg2pose/timeseries must be one dimensional")
    fields = dataset.dtype.fields or {}
    for name, shape in (("time", ()), ("emg", (16,)), ("joint_angles", (20,))):
        if name not in fields:
            raise ValueError(f"Missing timeseries field: {name}")
        dtype = fields[name][0]
        if dtype.shape != shape or not np.issubdtype(dtype.base, np.floating):
            raise ValueError(f"Unexpected dtype for {name}: {dtype}")


def _invalid_masks(rows: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ik_invalid_samples": np.all(np.isclose(rows["joint_angles"], 0), axis=1),
        "nonfinite_time_samples": ~np.isfinite(rows["time"]),
        "nonfinite_emg_samples": ~np.isfinite(rows["emg"]).all(axis=1),
        "nonfinite_angle_samples": ~np.isfinite(rows["joint_angles"]).all(axis=1),
    }


def valid_sample_mask(rows: np.ndarray) -> np.ndarray:
    """Official recomputed IK validity plus finite timestamps, EMG, and labels.

    Stored extra fields such as no_ik_failure are intentionally ignored. A single
    zero angle or an unprompted hand is not a reason to reject a sample.
    """
    return ~np.logical_or.reduce(list(_invalid_masks(rows).values()))


def scan_recording(
    recording: Recording,
    *,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    chunk_samples: int = 131_072,
) -> RecordingIndex:
    """Scan one file with bounded memory, without joining across invalid samples.

    Chunk size is an I/O setting, not an experimental window length. Segments
    crossing an I/O chunk boundary are joined only if genuinely continuous.
    """
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive and finite")
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    counts: dict[str, int | float] = {
        "samples": 0, "valid_samples": 0, "invalid_samples": 0,
        "ik_invalid_samples": 0, "nonfinite_time_samples": 0,
        "nonfinite_emg_samples": 0, "nonfinite_angle_samples": 0,
        "gap_edges": 0, "nonincreasing_time_edges": 0,
        "max_observed_gap_seconds": 0.0,
    }
    segments: list[Segment] = []
    source_stat = recording.path.stat()
    previous_time: float | None = None
    previous_valid = False
    with h5py.File(recording.path, "r") as file:
        dataset = file["emg2pose/timeseries"]
        _validate_schema(dataset)
        for offset in range(0, len(dataset), chunk_samples):
            rows = dataset[offset:offset + chunk_samples]
            times = rows["time"]
            invalid = _invalid_masks(rows)
            valid = ~np.logical_or.reduce(list(invalid.values()))
            dt = np.diff(times)
            linked = valid[:-1] & valid[1:] & (dt > 0) & (dt <= max_gap_seconds)
            starts = np.flatnonzero(valid & np.r_[True, ~linked])
            stops = np.flatnonzero(valid & np.r_[~linked, True]) + 1
            for start, stop in zip(starts, stops):
                global_start, global_stop = offset + int(start), offset + int(stop)
                predecessor_valid = previous_valid if start == 0 else bool(valid[start - 1])
                predecessor_time = previous_time if start == 0 else float(times[start - 1])
                gap = None if predecessor_time is None else float(times[start] - predecessor_time)
                if start == 0 and segments and predecessor_valid and 0 < gap <= max_gap_seconds:
                    segments[-1] = replace(
                        segments[-1], stop=global_stop, end_time=float(times[stop - 1]),
                    )
                    continue
                if global_start == 0:
                    reason = "recording_start"
                elif not predecessor_valid:
                    reason = "invalid_predecessor"
                else:
                    reason = "nonincreasing_time" if gap <= 0 else "timestamp_gap"
                segments.append(Segment(
                    global_start, global_stop, float(times[start]), float(times[stop - 1]), reason,
                ))
            edge_dt = dt if previous_time is None else np.r_[times[0] - previous_time, dt]
            finite_dt = edge_dt[np.isfinite(edge_dt)]
            counts["samples"] += len(rows)
            counts["valid_samples"] += int(valid.sum())
            counts["invalid_samples"] += int((~valid).sum())
            for key, mask in invalid.items():
                counts[key] += int(mask.sum())
            counts["gap_edges"] += int((finite_dt > max_gap_seconds).sum())
            counts["nonincreasing_time_edges"] += int((finite_dt <= 0).sum())
            counts["max_observed_gap_seconds"] = max(
                counts["max_observed_gap_seconds"], float(finite_dt.max(initial=0)),
            )
            previous_time, previous_valid = float(times[-1]), bool(valid[-1])
    counts["segments"] = len(segments)
    counts["valid_duration_seconds"] = sum(segment.duration_seconds for segment in segments)
    final_stat = recording.path.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
        raise ValueError("Source file changed while scanning; rescan it")
    return RecordingIndex(recording, max_gap_seconds, tuple(segments), counts,
                          source_stat.st_size, source_stat.st_mtime_ns)


def iter_indexes(index_path: Path, data_dir: Path = DEFAULT_DATA_DIR) -> Iterator[RecordingIndex]:
    """Reuse a saved JSONL index without repeating the signal scan."""
    with Path(index_path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            item = json.loads(line)
            if item.get("status") != "ok" or item.get("schema_version") != 1:
                raise ValueError(f"Index line {line_number} is an error record or unsupported schema")
            filename = item["filename"]
            if Path(filename).name != filename or "/" in filename or "\\" in filename:
                raise ValueError(f"Index line {line_number} contains an invalid filename")
            yield RecordingIndex(
                Recording(Path(data_dir) / filename, item["metadata"]),
                item["max_gap_seconds"], tuple(Segment(**s) for s in item["segments"]),
                item["statistics"], item["source_size"], item["source_mtime_ns"],
            )


def read_window(
    index: RecordingIndex,
    segment_id: int,
    *,
    offset: int = 0,
    length: int | None = None,
) -> dict[str, Any]:
    """Read raw arrays inside one valid segment, including any desired history.

    time: float64 [T] (original absolute timestamps), emg: [T, 16],
    joint_angles: radians [T, 20]. No resampling, normalization or zero padding.
    An explicit history/rollout division belongs to the later training sampler.
    """
    if not 0 <= segment_id < len(index.segments):
        raise IndexError("segment_id out of range")
    segment = index.segments[segment_id]
    length = segment.samples - offset if length is None else length
    if offset < 0 or length <= 0 or offset + length > segment.samples:
        raise ValueError("Requested window extends outside its valid segment")
    start, stop = segment.start + offset, segment.start + offset + length
    source_stat = index.recording.path.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (index.source_size, index.source_mtime_ns):
        raise ValueError("Source file changed since indexing; rescan it")
    # Opening per call avoids retaining/pickling shared h5py handles on Windows.
    with h5py.File(index.recording.path, "r") as file:
        rows = file["emg2pose/timeseries"][start:stop]
    dt = np.diff(rows["time"])
    if (len(rows) != length or not valid_sample_mask(rows).all()
            or np.any(dt <= 0) or np.any(dt > index.max_gap_seconds)):
        raise ValueError("Window no longer matches its validity index; rescan the source file")
    return {
        "time": rows["time"].copy(),
        "emg": rows["emg"].copy(),
        "joint_angles": rows["joint_angles"].copy(),
        "metadata": dict(index.recording.metadata),
        "source_filename": index.recording.path.name,
        "start": start, "stop": stop, "segment_id": segment_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-gap-ms", type=float, default=0.75,
                        help="Project gap rule, not an official Meta threshold")
    parser.add_argument("--output", type=Path, required=True, help="Local JSONL interval index")
    parser.add_argument("--limit", type=int, help="Bounded smoke check only; omit to scan every file")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent file scans (I/O setting)")
    args = parser.parse_args()
    if not np.isfinite(args.max_gap_ms) or args.max_gap_ms <= 0:
        parser.error("--max-gap-ms must be positive and finite")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.output.suffix != ".jsonl":
        parser.error("--output must have a .jsonl suffix")
    recordings = load_recordings(args.data_dir)
    selected = recordings if args.limit is None else recordings[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    totals: dict[str, int | float] = {}
    errors = 0

    def scan_one(recording: Recording) -> dict[str, Any]:
        try:
            result = scan_recording(recording, max_gap_seconds=args.max_gap_ms / 1000).to_dict()
            result["status"] = "ok"
            return result
        except (OSError, KeyError, ValueError) as error:
            return {"filename": recording.path.name, "status": "error", "error": str(error)}

    # Refuse to overwrite an existing index; failed files remain explicit records.
    with args.output.open("x", encoding="utf-8") as output, ThreadPoolExecutor(args.workers) as pool:
        # map preserves metadata order even if workers finish out of order.
        for number, result in enumerate(pool.map(scan_one, selected), start=1):
            if result["status"] == "ok":
                for key, value in result["statistics"].items():
                    if key == "max_observed_gap_seconds":
                        totals[key] = max(totals.get(key, 0), value)
                    else:
                        totals[key] = totals.get(key, 0) + value
            else:
                errors += 1
            output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            if number == 1 or number % 100 == 0 or number == len(selected):
                print(f"Scanned {number}/{len(selected)} files; errors={errors}", flush=True)
    summary = {
        "metadata_files": len(recordings), "scanned_files": len(selected), "errors": errors,
        "max_gap_seconds": args.max_gap_ms / 1000, "statistics": totals,
        "scope": "all_metadata" if args.limit is None else "smoke_check",
        "invalid_reason_counts_may_overlap": True,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
