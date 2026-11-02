"""Paths for the standalone fresh-training repository."""
from pathlib import Path
import os

CORE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = CORE_DIR.parent
PROJECT_DIR = EXPERIMENT_DIR
RESOURCES_DIR = CORE_DIR / 'resources'
HOD_ASSETS_DIR = EXPERIMENT_DIR / 'resources/robot'
BUNDLES_DIR = EXPERIMENT_DIR / 'experiments'
CACHE_DIR = EXPERIMENT_DIR / '.cache'
SHARED_DATA_DIR = EXPERIMENT_DIR / 'data'
DATA_DIR = Path(os.environ.get('EMG_DATA_DIR', EXPERIMENT_DIR / 'data/emg2pose_data')).expanduser().resolve()


def bundle_path(name: str, *parts: str | Path) -> Path:
    return BUNDLES_DIR.joinpath(name, *parts).resolve()


def resolve_experiment_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    relative = path.as_posix()
    if relative == 'data/emg2pose_data':
        return DATA_DIR
    if relative.startswith('experiment/'):
        relative = relative[len('experiment/'):]
    return (EXPERIMENT_DIR / relative).resolve()
