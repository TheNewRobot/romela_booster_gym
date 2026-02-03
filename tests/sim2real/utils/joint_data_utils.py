"""
Additional data utilities for joint sysid data.

Add these functions to tests/sim2real/utils/data_utils.py
"""

import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class JointData:
    """Container for a single joint's test data."""
    joint_idx: int
    joint_name: str
    timestamps: np.ndarray
    cmd_positions: np.ndarray
    actual_positions: np.ndarray
    actual_velocities: np.ndarray
    actual_torques: np.ndarray
    kp: float
    kd: float


def parse_csv_header(filepath: str) -> dict:
    """Extract metadata from CSV header comments."""
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                parts = line[1:].strip().split(':', 1)
                if len(parts) == 2:
                    metadata[parts[0].strip()] = parts[1].strip()
            else:
                break
    return metadata


def load_joint_data(csv_path: str, robot_cfg: dict) -> JointData:
    """
    Load joint test data from CSV file.
    
    Args:
        csv_path: Path to joint CSV file
        robot_cfg: Robot config dict (T1.yaml contents)
    
    Returns:
        JointData object with all test data
    """
    metadata = parse_csv_header(csv_path)
    df = pd.read_csv(csv_path, comment='#')
    
    joint_idx = int(metadata.get('joint_index', -1))
    joint_name = metadata.get('joint_name', 'Unknown')
    
    # Normalize timestamps to start at 0
    timestamps = df['timestamp'].values - df['timestamp'].values[0]
    
    # Get PD gains from robot config
    kp = robot_cfg['common']['stiffness'][joint_idx]
    kd = robot_cfg['common']['damping'][joint_idx]
    
    return JointData(
        joint_idx=joint_idx,
        joint_name=joint_name,
        timestamps=timestamps.astype(np.float32),
        cmd_positions=df['cmd_position'].values.astype(np.float32),
        actual_positions=df['actual_position'].values.astype(np.float32),
        actual_velocities=df['actual_velocity'].values.astype(np.float32),
        actual_torques=df['actual_torque'].values.astype(np.float32),
        kp=kp,
        kd=kd,
    )


def load_all_joint_data(
    experiment_dir: Path,
    robot_cfg: dict,
    verbose: bool = True,
) -> Dict[int, JointData]:
    """
    Load all joint CSV files from experiment directory.
    
    Args:
        experiment_dir: Path to experiment folder
        robot_cfg: Robot config dict
        verbose: Print loading info
    
    Returns:
        Dict mapping joint_idx to JointData
    """
    csv_files = sorted(glob.glob(str(experiment_dir / 'joint_*.csv')))
    joint_data = {}
    
    for csv_path in csv_files:
        try:
            data = load_joint_data(csv_path, robot_cfg)
            joint_data[data.joint_idx] = data
            if verbose:
                print(f"  Loaded joint {data.joint_idx}: {data.joint_name} "
                      f"({len(data.timestamps)} samples)")
        except Exception as e:
            if verbose:
                print(f"  Error loading {csv_path}: {e}")
    
    return joint_data