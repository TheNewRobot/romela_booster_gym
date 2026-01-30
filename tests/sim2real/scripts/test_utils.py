import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from utils.data_utils import (
    load_config, save_csv, load_csv, get_experiment_path,
    create_experiment_dirs, load_experiment, profile_to_timeseries_v2
)


def test_config_loading():
    print(f"\n{'='*60}")
    print("Testing Config Loading")
    print(f"{'='*60}")

    configs = ['topics', 'command_profiles', 'comparison', 'sim_params']
    for name in configs:
        try:
            cfg = load_config(name)
            if name == 'topics':
                keys = list(cfg.keys())
                print(f"[✓] {name}.yaml — keys: {', '.join(keys)}")
            elif name == 'command_profiles':
                profiles = list(cfg['profiles'].keys())
                print(f"[✓] {name}.yaml — profiles: {', '.join(profiles)}")
            elif name == 'comparison':
                n_metrics = len(cfg['metrics'])
                n_plots = len(cfg['plots'])
                print(f"[✓] {name}.yaml — metrics: {n_metrics}, plots: {n_plots}")
            elif name == 'sim_params':
                sections = list(cfg.keys())
                print(f"[✓] {name}.yaml — sections: {', '.join(sections)}")
        except Exception as e:
            print(f"[✗] {name}.yaml — {e}")
            return False
    return True


def test_csv_roundtrip():
    print(f"\n{'='*60}")
    print("Testing CSV Round-Trip")
    print(f"{'='*60}")

    test_dir = Path(__file__).parent.parent / "data" / "_test_temp"
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Test joint_states format
        n_joints = 23
        df_joints = pd.DataFrame({
            'timestamp': np.linspace(0, 1, 50),
            **{f'q_{i}': np.random.randn(50) for i in range(n_joints)},
            **{f'dq_{i}': np.random.randn(50) for i in range(n_joints)},
        })
        path_joints = test_dir / "joint_states.csv"
        save_csv(df_joints, path_joints)
        df_loaded = load_csv(path_joints)
        assert df_joints.shape == df_loaded.shape, "Shape mismatch"
        assert np.allclose(df_joints.values, df_loaded.values), "Values mismatch"
        print(f"[✓] CSV round-trip passed (joint_states: {df_joints.shape})")

        # Test base_state format
        df_base = pd.DataFrame({
            'timestamp': np.linspace(0, 1, 50),
            'x': np.random.randn(50), 'y': np.random.randn(50), 'z': np.random.randn(50),
            'qx': np.random.randn(50), 'qy': np.random.randn(50),
            'qz': np.random.randn(50), 'qw': np.random.randn(50),
            'vx': np.random.randn(50), 'vy': np.random.randn(50), 'vz': np.random.randn(50),
            'wx': np.random.randn(50), 'wy': np.random.randn(50), 'wz': np.random.randn(50),
        })
        path_base = test_dir / "base_state.csv"
        save_csv(df_base, path_base)
        df_loaded = load_csv(path_base)
        assert df_base.shape == df_loaded.shape, "Shape mismatch"
        print(f"[✓] CSV round-trip passed (base_state: {df_base.shape})")

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

    return True


def test_experiment_paths():
    print(f"\n{'='*60}")
    print("Testing Experiment Path Creation")
    print(f"{'='*60}")

    test_exp = "_test_experiment"
    try:
        dirs = create_experiment_dirs(test_exp)
        expected = ['root', 'real', 'isaac', 'mujoco', 'plots']
        for key in expected:
            assert key in dirs, f"Missing key: {key}"
            assert dirs[key].exists(), f"Directory not created: {dirs[key]}"
        print(f"[✓] Experiment directories created: {list(dirs.keys())}")
    finally:
        shutil.rmtree(get_experiment_path(test_exp), ignore_errors=True)

    return True


def test_command_profile():
    print(f"\n{'='*60}")
    print("Testing Command Profile Generation")
    print(f"{'='*60}")

    cfg = load_config('command_profiles')
    dt = cfg['defaults']['control_dt']

    for profile_name in ['stand', 'walk_forward', 'mixed']:
        profile = cfg['profiles'][profile_name]
        df = profile_to_timeseries_v2(profile, dt)

        total_duration = sum(seg['duration'] for seg in profile)
        expected_rows = int(total_duration / dt)

        assert len(df) == expected_rows, f"Row count mismatch: {len(df)} vs {expected_rows}"
        assert list(df.columns) == ['timestamp', 'vx', 'vy', 'vyaw'], "Column mismatch"
        assert df['timestamp'].iloc[0] == 0.0, "Should start at t=0"

        print(f"[✓] Profile '{profile_name}' → {len(df)} rows, {total_duration}s duration")

    return True


def main():
    print(f"\n{'='*60}")
    print("Sim2Real Utilities Test Suite")
    print(f"{'='*60}")

    all_passed = True
    all_passed &= test_config_loading()
    all_passed &= test_csv_roundtrip()
    all_passed &= test_experiment_paths()
    all_passed &= test_command_profile()

    print(f"\n{'='*60}")
    if all_passed:
        print("All tests passed ✓")
    else:
        print("Some tests failed ✗")
    print(f"{'='*60}\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())