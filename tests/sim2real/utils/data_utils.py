import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


CONFIG_DIR = Path(__file__).parent.parent / "config"
DATA_DIR = Path(__file__).parent.parent / "data"


def load_yaml(path: Path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_config(name: str) -> dict:
    return load_yaml(CONFIG_DIR / f"{name}.yaml")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def get_experiment_path(exp_name: str) -> Path:
    return DATA_DIR / exp_name


def create_experiment_dirs(exp_name: str) -> Dict[str, Path]:
    exp_path = get_experiment_path(exp_name)
    dirs = {
        'root': exp_path,
        'real': exp_path / 'real',
        'isaac': exp_path / 'isaac',
        'mujoco': exp_path / 'mujoco',
        'plots': exp_path / 'plots',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_experiment(exp_name: str, source: str) -> Dict[str, Optional[pd.DataFrame]]:
    """Load all CSV files for an experiment from a given source (real/isaac/mujoco)."""
    source_dir = get_experiment_path(exp_name) / source
    data = {}
    for csv_name in ['commands', 'joint_states', 'base_state']:
        csv_path = source_dir / f"{csv_name}.csv"
        data[csv_name] = load_csv(csv_path) if csv_path.exists() else None
    return data


def profile_to_timeseries(profile: list, dt: float) -> pd.DataFrame:
    """Convert a command profile (list of segments) to a timestamped DataFrame."""
    rows = []
    t = 0.0
    for seg in profile:
        duration = seg['duration']
        while t < sum(s['duration'] for s in profile[:profile.index(seg)]) + duration:
            rows.append({'timestamp': t, 'vx': seg['vx'], 'vy': seg['vy'], 'vyaw': seg['vyaw']})
            t += dt
    return pd.DataFrame(rows)


def profile_to_timeseries_v2(profile: list, dt: float) -> pd.DataFrame:
    """Convert a command profile to a timestamped DataFrame (cleaner implementation)."""
    rows = []
    t = 0.0
    for seg in profile:
        n_steps = int(seg['duration'] / dt)
        for _ in range(n_steps):
            rows.append({'timestamp': round(t, 6), 'vx': seg['vx'], 'vy': seg['vy'], 'vyaw': seg['vyaw']})
            t += dt
    return pd.DataFrame(rows)