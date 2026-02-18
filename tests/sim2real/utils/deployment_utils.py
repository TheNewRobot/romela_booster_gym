"""Utilities for loading and processing deployment and Vicon data."""

import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from scipy.interpolate import interp1d


@dataclass
class DeploymentData:
    """Container for deployment CSV data."""
    timestamps: np.ndarray          # (N,) relative seconds
    vx: np.ndarray                  # (N,)
    vy: np.ndarray                  # (N,)
    vyaw: np.ndarray                # (N,)
    q_act: np.ndarray               # (N, num_joints)
    dq_act: np.ndarray              # (N, num_joints)
    tau_est: np.ndarray             # (N, num_joints)
    q_cmd: np.ndarray               # (N, num_joints)
    actions: np.ndarray             # (N, num_actions)
    imu: np.ndarray                 # (N, 9) roll,pitch,yaw,gyro_xyz,acc_xyz
    policy_name: str = ""
    profile_name: str = ""


@dataclass
class ViconData:
    """Container for Vicon motion capture data."""
    wall_time: np.ndarray           # (M,) unix timestamps
    base_pos: np.ndarray            # (M, 3) x,y,z
    base_quat: np.ndarray           # (M, 4) qx,qy,qz,qw


def load_deployment_data(csv_path: str, cfg: dict) -> DeploymentData:
    """Load deployment CSV into DeploymentData, using cfg for array dimensions."""
    csv_path = Path(csv_path)
    num_joints = cfg["joints"]["total"]
    num_actions = cfg["env"]["num_actions"]

    # Parse metadata from comment lines
    policy_name = ""
    profile_name = ""
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('#'):
                break
            if line.startswith('# policy:'):
                policy_name = line.split(':', 1)[1].strip()
            elif line.startswith('# profile:'):
                profile_name = line.split(':', 1)[1].strip()

    df = pd.read_csv(csv_path, comment='#')

    timestamps = df['timestamp'].values
    timestamps = timestamps - timestamps[0]

    q_act = np.column_stack([df[f'q_act_{i}'].values for i in range(num_joints)])
    dq_act = np.column_stack([df[f'dq_act_{i}'].values for i in range(num_joints)])
    tau_est = np.column_stack([df[f'tau_est_{i}'].values for i in range(num_joints)])
    q_cmd = np.column_stack([df[f'q_cmd_{i}'].values for i in range(num_joints)])
    actions = np.column_stack([df[f'action_{i}'].values for i in range(num_actions)])

    imu = np.column_stack([
        df['roll'].values, df['pitch'].values, df['yaw'].values,
        df['gyro_x'].values, df['gyro_y'].values, df['gyro_z'].values,
        df['acc_x'].values, df['acc_y'].values, df['acc_z'].values,
    ])

    return DeploymentData(
        timestamps=timestamps,
        vx=df['vx'].values, vy=df['vy'].values, vyaw=df['vyaw'].values,
        q_act=q_act, dq_act=dq_act, tau_est=tau_est, q_cmd=q_cmd,
        actions=actions, imu=imu,
        policy_name=policy_name, profile_name=profile_name,
    )


def load_vicon_data(csv_path: str) -> ViconData:
    """Load Vicon CSV (wall_time, base_x/y/z, base_qx/qy/qz/qw)."""
    df = pd.read_csv(csv_path)
    return ViconData(
        wall_time=df['wall_time'].values,
        base_pos=np.column_stack([
            df['base_x'].values, df['base_y'].values, df['base_z'].values,
        ]),
        base_quat=np.column_stack([
            df['base_qx'].values, df['base_qy'].values,
            df['base_qz'].values, df['base_qw'].values,
        ]),
    )


def find_vicon_data(deploy_dir: str) -> Optional[str]:
    """Auto-detect vicon_log.csv in the deployment folder or vicon_*/ subdirs."""
    deploy_dir = Path(deploy_dir)

    direct = deploy_dir / 'vicon_log.csv'
    if direct.exists():
        return str(direct)

    matches = sorted(glob.glob(str(deploy_dir / 'vicon_*' / 'vicon_log.csv')))
    if matches:
        return matches[-1]

    return None


def align_timestamps(
    deploy: DeploymentData,
    vicon: ViconData,
    search_step: float = 0.02,
) -> Tuple[float, np.ndarray]:
    """Align Vicon timestamps to deployment time frame via cross-correlation.

    Finds the time offset that maximizes the match between the commanded
    forward velocity (|vx|) and the Vicon-measured forward speed.

    The offset represents: deployment_time + offset = vicon_relative_time.
    """
    # Compute Vicon speed from position derivatives
    vt_rel = vicon.wall_time - vicon.wall_time[0]
    dt_vicon = np.diff(vicon.wall_time)
    dx = np.diff(vicon.base_pos, axis=0)
    speed = np.linalg.norm(dx[:, :2], axis=1) / np.maximum(dt_vicon, 1e-6)
    speed_t = (vt_rel[:-1] + vt_rel[1:]) / 2

    kernel = np.ones(10) / 10
    if len(speed) > 10:
        speed = np.convolve(speed, kernel, mode='same')

    speed_interp = interp1d(speed_t, speed, bounds_error=False, fill_value=0)

    # Search range: offset must keep deployment window inside vicon data
    deploy_dur = deploy.timestamps[-1]
    vicon_dur = vt_rel[-1]
    off_min = max(0, -2.0)
    off_max = max(0, vicon_dur - deploy_dur + 2.0)

    # Find offset that maximizes dot product (rewards matching movement periods)
    cmd_speed = np.abs(deploy.vx)
    best_score, best_offset = -1, 0
    for off in np.arange(off_min, off_max, search_step):
        vicon_speed_resampled = speed_interp(deploy.timestamps + off)
        score = np.dot(cmd_speed, vicon_speed_resampled)
        if score > best_score:
            best_score, best_offset = score, off

    offset = vicon.wall_time[0] + best_offset
    vicon_times_aligned = vicon.wall_time - offset

    return best_offset, vicon_times_aligned


def resample_vicon_to_deployment(
    deploy: DeploymentData,
    vicon: ViconData,
    vicon_times_aligned: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample Vicon data to deployment timestamps via linear interpolation."""
    t_min = vicon_times_aligned[0]
    t_max = vicon_times_aligned[-1]
    t_deploy = np.clip(deploy.timestamps, t_min, t_max)

    pos_interp = interp1d(vicon_times_aligned, vicon.base_pos, axis=0,
                          kind='linear', fill_value='extrapolate')
    pos = pos_interp(t_deploy)

    quat_interp = interp1d(vicon_times_aligned, vicon.base_quat, axis=0,
                           kind='linear', fill_value='extrapolate')
    quat = quat_interp(t_deploy)
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.maximum(norms, 1e-8)

    return pos, quat


def transform_positions_to_body_frame(
    pos: np.ndarray,
    initial_quat: np.ndarray,
    quat_format: str = "xyzw",
) -> np.ndarray:
    """Transform positions to robot's initial body frame (origin, heading=+X)."""
    if quat_format == "xyzw":
        qx, qy, qz, qw = initial_quat
    else:
        qw, qx, qy, qz = initial_quat

    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    rel = pos - pos[0]
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)

    result = rel.copy()
    result[:, 0] = rel[:, 0] * cos_y - rel[:, 1] * sin_y
    result[:, 1] = rel[:, 0] * sin_y + rel[:, 1] * cos_y

    return result
