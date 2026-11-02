"""Generate a user-disjoint bilateral split and valid-interval index locally."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/rl_meta_pose_feedback'))
from _bootstrap import activate
activate()
from emg_control_paths import DATA_DIR
from rl_meta_pose_feedback.training.data import make_split, index_recordings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR,
                        help='Directory containing metadata.csv and the original HDF5 recordings')
    parser.add_argument('--output', type=Path, default=ROOT / 'data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.output.resolve()
    manifest_path = output / 'both_split.json'
    index_path = output / 'both_intervals.jsonl'
    if manifest_path.exists() or index_path.exists():
        parser.error('Refusing to replace an existing split/index; choose a new --output directory')
    manifest = make_split(args.data_dir.resolve(), seed=args.seed, sides=('left', 'right'))
    output.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    summary = index_recordings(manifest, index_path, data_dir=args.data_dir.resolve(), workers=args.workers)
    print(json.dumps(summary, indent=2))
    print('Use the same EMG_DATA_DIR for training and evaluation. Keep the recording files unchanged after indexing.')


if __name__ == '__main__':
    main()
