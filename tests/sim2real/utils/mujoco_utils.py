"""
MuJoCo utilities for sim2real comparison and optimization.

Provides functions to:
- Load and configure MuJoCo models
- Apply sim_params.yaml values at runtime
- Simulate joint responses
- Map joint indices to DOF indices
"""

import numpy as np
import mujoco
from typing import Dict, Tuple, Optional
from pathlib import Path


# Joint index mapping (leg joints for T1 locomotion)
LEG_JOINT_INDICES = list(range(11, 23))
LEG_JOINT_NAMES = {
    11: "Left_Hip_Pitch", 12: "Left_Hip_Roll", 13: "Left_Hip_Yaw",
    14: "Left_Knee_Pitch", 15: "Left_Ankle_Pitch", 16: "Left_Ankle_Roll",
    17: "Right_Hip_Pitch", 18: "Right_Hip_Roll", 19: "Right_Hip_Yaw",
    20: "Right_Knee_Pitch", 21: "Right_Ankle_Pitch", 22: "Right_Ankle_Roll",
}


def load_mujoco_model(xml_path: str) -> mujoco.MjModel:
    """Load MuJoCo model from XML file."""
    return mujoco.MjModel.from_xml_path(xml_path)


def get_joint_dof_indices(mj_model: mujoco.MjModel) -> Dict[int, Tuple[int, int]]:
    """
    Map T1 joint indices (11-22) to MuJoCo qpos and DOF indices.
    
    Returns dict: {joint_idx: (qpos_idx, dof_idx)}
    
    Note: qpos and dof indices differ when there's a floating base.
    - qpos_idx: index into mj_data.qpos (position)
    - dof_idx: index into mj_data.qvel, qfrc_applied (velocity/force)
    """
    joint_mapping = {}
    
    for i in range(mj_model.njnt):
        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name:
            # Match joint names to our indices
            for idx, name in LEG_JOINT_NAMES.items():
                # Flexible matching: remove underscores and compare lowercase
                if name.lower().replace('_', '') in joint_name.lower().replace('_', ''):
                    qpos_adr = mj_model.jnt_qposadr[i]
                    dof_adr = mj_model.jnt_dofadr[i]
                    joint_mapping[idx] = (qpos_adr, dof_adr)
                    break
    
    return joint_mapping


def apply_sim_params(
    mj_model: mujoco.MjModel,
    sim_params: dict,
    joint_mapping: Dict[int, Tuple[int, int]],
) -> None:
    """
    Apply sim_params.yaml values to MuJoCo model in-place.
    
    Args:
        mj_model: MuJoCo model to modify
        sim_params: Loaded sim_params.yaml dict
        joint_to_dof: Mapping from joint index to DOF index
    """
    mujoco_params = sim_params.get('mujoco', {}).get('joint', {})
    
    damping = mujoco_params.get('damping', {})
    armature = mujoco_params.get('armature', {})
    frictionloss = mujoco_params.get('frictionloss', {})
    
    for joint_idx, (qpos_idx, dof_idx) in joint_mapping.items():
        if isinstance(damping, dict) and joint_idx in damping:
            mj_model.dof_damping[dof_idx] = damping[joint_idx]
        elif isinstance(damping, (int, float)):
            mj_model.dof_damping[dof_idx] = damping
            
        if isinstance(armature, dict) and joint_idx in armature:
            mj_model.dof_armature[dof_idx] = armature[joint_idx]
        elif isinstance(armature, (int, float)):
            mj_model.dof_armature[dof_idx] = armature
            
        if isinstance(frictionloss, dict) and joint_idx in frictionloss:
            mj_model.dof_frictionloss[dof_idx] = frictionloss[joint_idx]
        elif isinstance(frictionloss, (int, float)):
            mj_model.dof_frictionloss[dof_idx] = frictionloss


def simulate_joint_response(
    mj_model: mujoco.MjModel,
    joint_indices: Tuple[int, int],
    cmd_positions: np.ndarray,
    kp: float,
    kd: float,
    initial_pos: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate joint response to commanded positions using PD control.
    
    Args:
        mj_model: MuJoCo model (with params already applied)
        joint_indices: Tuple of (qpos_idx, dof_idx) from get_joint_dof_indices
        cmd_positions: Array of commanded positions
        kp: Proportional gain
        kd: Derivative gain
        initial_pos: Initial joint position (default: first cmd)
    
    Returns:
        sim_positions: Simulated position trajectory
        sim_velocities: Simulated velocity trajectory
    """
    qpos_idx, dof_idx = joint_indices
    
    data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, data)
    
    # Fix base in hanging position (robot suspended in air)
    base_pos = np.array([0.0, 0.0, 1.0])  # 1m above ground
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])  # upright (w,x,y,z)
    data.qpos[0:3] = base_pos
    data.qpos[3:7] = base_quat
    data.qvel[0:6] = 0.0  # no base velocity
    
    # Set initial position (use qpos_idx)
    if initial_pos is not None:
        data.qpos[qpos_idx] = initial_pos
    else:
        data.qpos[qpos_idx] = cmd_positions[0]
    
    mujoco.mj_forward(mj_model, data)
    
    n_steps = len(cmd_positions)
    sim_positions = np.zeros(n_steps, dtype=np.float32)
    sim_velocities = np.zeros(n_steps, dtype=np.float32)
    
    for i, cmd_pos in enumerate(cmd_positions):
        # PD control (qpos for position, qvel for velocity)
        pos_error = cmd_pos - data.qpos[qpos_idx]
        vel = data.qvel[dof_idx]
        tau = kp * pos_error - kd * vel
        
        # Apply torque (qfrc_applied uses dof_idx)
        data.qfrc_applied[dof_idx] = tau
        
        # Step
        mujoco.mj_step(mj_model, data)
        
        # Keep base fixed (simulate hanging robot)
        data.qpos[0:3] = base_pos
        data.qpos[3:7] = base_quat
        data.qvel[0:6] = 0.0
        
        # Record
        sim_positions[i] = data.qpos[qpos_idx]
        sim_velocities[i] = data.qvel[dof_idx]
    
    return sim_positions, sim_velocities


def simulate_joint_with_params(
    mj_model: mujoco.MjModel,
    joint_indices: Tuple[int, int],
    cmd_positions: np.ndarray,
    kp: float,
    kd: float,
    damping: float,
    armature: float,
    frictionloss: float,
    initial_pos: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate joint response with specific dynamics parameters.
    
    Temporarily modifies model params, simulates, then could restore
    (but we use a fresh model each time in optimization).
    """
    qpos_idx, dof_idx = joint_indices
    
    # Apply params (dof_* arrays use dof_idx)
    mj_model.dof_damping[dof_idx] = damping
    mj_model.dof_armature[dof_idx] = armature
    mj_model.dof_frictionloss[dof_idx] = frictionloss
    
    return simulate_joint_response(
        mj_model, joint_indices, cmd_positions, kp, kd, initial_pos
    )


def compute_metrics(
    sim_positions: np.ndarray,
    real_positions: np.ndarray,
    sim_velocities: Optional[np.ndarray] = None,
    real_velocities: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comparison metrics between sim and real trajectories.
    
    Returns dict with: rmse_pos, rmse_vel, max_error_pos, correlation
    """
    pos_error = sim_positions - real_positions
    
    metrics = {
        'rmse_pos': float(np.sqrt(np.mean(pos_error ** 2))),
        'max_error_pos': float(np.max(np.abs(pos_error))),
        'mean_error_pos': float(np.mean(np.abs(pos_error))),
    }
    
    # Correlation
    if len(sim_positions) > 1:
        corr = np.corrcoef(sim_positions, real_positions)[0, 1]
        metrics['correlation'] = float(corr) if not np.isnan(corr) else 0.0
    
    # Velocity metrics if available
    if sim_velocities is not None and real_velocities is not None:
        vel_error = sim_velocities - real_velocities
        metrics['rmse_vel'] = float(np.sqrt(np.mean(vel_error ** 2)))
    
    return metrics