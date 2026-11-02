"""Experiment-owned bilateral splits and validated interval indexes.

The recordings module owns signal validity rules. Paths remain anchored
to the experiment root, independently of the working directory.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from data_pipeline.recordings import Recording, RecordingIndex, iter_indexes, load_recordings, scan_recording


from emg_control_paths import DATA_DIR

DEFAULT_DATA_DIR = DATA_DIR
SPLITS = ("train", "val", "test")


def make_split(data_dir: Path = DEFAULT_DATA_DIR, *, seed: int = 42, sides=("left", "right")) -> dict:
    """Split selected hand sides together by user, approximately 80/10/10 users.

    All stages and moving_hand values are retained; technical validity is a
    separate indexing step. Original official split metadata is preserved.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    sides = tuple(sorted(sides))
    if not sides or len(set(sides)) != len(sides) or not set(sides) <= {"left", "right"}:
        raise ValueError("sides must select left, right, or both without duplicates")
    recordings = [r for r in load_recordings(data_dir) if r.metadata["side"] in sides]
    users = sorted({r.metadata["user"] for r in recordings})
    if len(users) < 3:
        raise ValueError("Need at least three users for three disjoint splits")
    # Hash ordering is stable across Python versions and metadata row order.
    users.sort(key=lambda user: hashlib.sha256(f"{seed}:{user}".encode()).digest())
    held_out = max(1, len(users) // 10)
    assignments = {user: "test" if i < held_out else "val" if i < 2 * held_out else "train"
                   for i, user in enumerate(users)}
    records = [{"filename": r.path.name, "split": assignments[r.metadata["user"]],
                "metadata": r.metadata} for r in sorted(recordings, key=lambda r: r.path.name)]
    return {
        "schema_version": 2, "sides": list(sides), "group_by": "user", "seed": seed,
        "requested_user_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "metadata_sha256": hashlib.sha256((Path(data_dir) / "metadata.csv").read_bytes()).hexdigest(),
        "summary": {split: {
            "users": sum(value == split for value in assignments.values()),
            "recordings": sum(r["split"] == split for r in records),
            "by_side": {side: sum(r["split"] == split and r["metadata"]["side"] == side for r in records)
                        for side in sides},
        } for split in SPLITS},
        "records": records,
    }


def load_split_manifest(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") == 1 and manifest.get("side") == "right":
        sides = {"right"}  # Saved right-only checkpoints retain their original pools.
    elif manifest.get("schema_version") == 2:
        values = manifest.get("sides", [])
        sides = set(values)
        if not sides or len(sides) != len(values) or not sides <= {"left", "right"}:
            raise ValueError("Manifest has invalid hand sides")
    else:
        raise ValueError("Unsupported hand-side split manifest")
    if manifest.get("group_by") != "user":
        raise ValueError("Manifest must group both hands by user")
    seen, user_split = set(), {}
    for record in manifest["records"]:
        name, split, metadata = record["filename"], record["split"], record["metadata"]
        if (Path(name).name != name or "/" in name or "\\" in name or name in seen
                or split not in SPLITS or metadata.get("side") not in sides):
            raise ValueError("Manifest has invalid, duplicate, or unselected-side recordings")
        seen.add(name)
        user = metadata["user"]
        if user in user_split and user_split[user] != split:
            raise ValueError("One user appears in multiple experiment splits")
        user_split[user] = split
    if not seen or set(user_split.values()) != set(SPLITS):
        raise ValueError("Manifest must contain all three nonempty splits")
    return manifest


def index_recordings(manifest: dict, output: Path, *, data_dir: Path = DEFAULT_DATA_DIR,
                           workers: int = 4, max_gap_seconds: float = 0.00075, reuse_index: Path | None = None) -> dict:
    """Index every manifest recording once; report failures instead of dropping them."""
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be positive and finite")
    output = Path(output)
    if output.suffix != ".jsonl":
        raise ValueError("Interval index must use .jsonl")
    records = manifest["records"]
    reusable = {} if reuse_index is None else {
        index.recording.path.name: index for index in iter_indexes(reuse_index, data_dir=data_dir)}

    def scan(record: dict) -> dict:
        metadata = dict(record["metadata"], experiment_split=record["split"])
        try:
            path = Path(data_dir) / record["filename"]
            old = reusable.get(record["filename"])
            stat = path.stat()
            if (old is not None and old.max_gap_seconds == max_gap_seconds
                    and (old.source_size, old.source_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
                    and all(old.recording.metadata.get(k) == v for k, v in record["metadata"].items())):
                item = old.to_dict()
                item["metadata"] = metadata
                return dict(item, status="ok", reused=True)
            index = scan_recording(Recording(path, metadata), max_gap_seconds=max_gap_seconds)
            return dict(index.to_dict(), status="ok", reused=False)
        except (OSError, KeyError, ValueError) as error:
            return {"filename": record["filename"], "status": "error", "error": str(error)}

    output.parent.mkdir(parents=True, exist_ok=True)
    totals = Counter({"recordings": 0, "errors": 0})
    with output.open("x", encoding="utf-8") as file, ThreadPoolExecutor(workers) as pool:
        for number, item in enumerate(pool.map(scan, records), 1):
            file.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
            if item["status"] == "ok":
                totals["recordings"] += 1
                totals["reused_indexes"] += int(item["reused"])
                for key in ("samples", "valid_samples", "invalid_samples", "segments",
                            "valid_duration_seconds", "gap_edges", "nonincreasing_time_edges"):
                    totals[key] += item["statistics"][key]
            else:
                totals["errors"] += 1
            if number == 1 or number % 100 == 0 or number == len(records):
                file.flush()
                print(f"Hand index: {number}/{len(records)}, errors={totals['errors']}", flush=True)
    summary = {"scope": "selected_hand_recordings", "sides": manifest.get("sides", ["right"]), "split_by": "user",
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
        if (metadata.get("side") != record["metadata"]["side"] or metadata.get("user") != record["metadata"]["user"]
                or metadata.get("experiment_split") != record["split"]):
            raise ValueError("Interval index does not match the split manifest")
        seen.add(name)
        if record["split"] == split:
            selected.append(index)
    if seen != set(expected):
        raise ValueError("Interval index is incomplete; refusing a partial training split")
    return selected
