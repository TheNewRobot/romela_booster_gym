"""
Policy Arena Utilities

Shared functions for policy evaluation in both Isaac Gym and MuJoCo.
"""
import os
import csv
import yaml
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import torch


def load_yaml(path: str) -> dict:
    """Load YAML file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arena_config(config: dict, headless_override: Optional[bool] = None) -> dict:
    """
    Parse arena config into structured format.
    
    Returns dict with: policies, evaluation, timing, terrains, stages
    """
    parsed = {
        'policies': config['policies'],
        'terrains': config['terrains'],
        'stages': config['stages'],
    }
    
    eval_cfg = config['evaluation'].copy()
    if headless_override is not None:
        eval_cfg['headless'] = headless_override
    parsed['evaluation'] = eval_cfg
    
    parsed['timing'] = config['timing'].copy()
    
    return parsed


def compute_command_ramp(t: float, settle_s: float, ramp_s: float, target: float) -> float:
    """
    Compute ramped command value.
    
    Timeline: [0, settle_s] = 0, [settle_s, settle_s+ramp_s] = linear ramp, after = target
    """
    if t < settle_s:
        return 0.0
    elif t < settle_s + ramp_s:
        alpha = (t - settle_s) / ramp_s
        return target * alpha
    else:
        return target


def get_stage_commands(stage: dict, t: float, timing: dict) -> Tuple[float, float, float]:
    """Get velocity commands for current time in stage."""
    settle_s = timing['settle_duration_s']
    ramp_s = timing['ramp_duration_s']
    
    vx = compute_command_ramp(t, settle_s, ramp_s, stage['vx'])
    vy = compute_command_ramp(t, settle_s, ramp_s, stage['vy'])
    vyaw = compute_command_ramp(t, settle_s, ramp_s, stage['vyaw'])
    
    return vx, vy, vyaw


def check_survival(base_height: float, min_height: float = 0.3) -> bool:
    """Check if robot has fallen (base height too low)."""
    return base_height > min_height


def compute_tracking_error(actual_vel: np.ndarray, target_vel: np.ndarray) -> float:
    """Compute velocity tracking error (L2 norm)."""
    return float(np.linalg.norm(actual_vel - target_vel))


def check_stage_pass(
    survival_rate: float,
    survival_threshold: float,
    tracking_error: Optional[float] = None,
    tracking_threshold: Optional[float] = None
) -> bool:
    """
    Check if stage is passed.
    
    Must meet survival threshold, and tracking threshold if specified.
    """
    if survival_rate < survival_threshold:
        return False
    if tracking_threshold is not None and tracking_error is not None:
        if tracking_error > tracking_threshold:
            return False
    return True


def create_output_dir(task: str, base_dir: str = "exps") -> Tuple[str, str]:
    """
    Create timestamped output directory.
    
    Returns (output_dir, timestamp)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join(base_dir, task, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, timestamp


def save_results_csv(results: List[dict], output_dir: str, filename: str = "results.csv"):
    """Save results list to CSV file."""
    if not results:
        print("[!] No results to save.")
        return
    
    csv_path = os.path.join(output_dir, filename)
    fieldnames = list(results[0].keys())
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"[✓] Results saved to: {csv_path}")


def print_stage_result(
    stage_idx: int,
    stage_name: str,
    policy_name: str,
    survival_rate: float,
    tracking_error: Optional[float],
    passed: bool,
    survival_threshold: float,
    tracking_threshold: Optional[float] = None,
    suffix: str = ""
):
    """Print formatted stage result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    
    tracking_str = ""
    if tracking_error is not None:
        thresh_str = f"/{tracking_threshold:.3f}" if tracking_threshold else ""
        tracking_str = f"  TrackErr: {tracking_error:.3f}{thresh_str}{suffix}"
    
    print(f"  [{policy_name}] Stage {stage_idx+1} '{stage_name}': "
          f"Survival: {survival_rate:.1%}/{survival_threshold:.0%}{tracking_str}  {status}")


def print_arena_header(
    policies: List[dict],
    num_trials: int,
    num_stages: int,
    stage_duration: float,
    task: str,
    output_dir: str,
    headless: bool,
    sim_params: str = "default"
):
    """Print evaluation header."""
    width = 60
    print(f"\n{'='*width}")
    print("MUJOCO POLICY ARENA EVALUATION".center(width))
    print(f"{'='*width}")
    print(f"  Policies:       {[p['name'] for p in policies]}")
    print(f"  Trials/policy:  {num_trials}")
    print(f"  Stages:         {num_stages}")
    print(f"  Stage duration: {stage_duration}s")
    print(f"  Task:           {task}")
    print(f"  Sim params:     {sim_params}")
    print(f"  Headless:       {headless}")
    print(f"  Output:         {output_dir}")
    print(f"{'='*width}\n")


def print_arena_summary(policy_status: OrderedDict, num_stages: int, output_dir: str):
    """Print final summary of results."""
    width = 60
    
    print(f"\n{'='*width}")
    print("FINAL RESULTS".center(width))
    print(f"{'='*width}")
    
    for policy_name, status in policy_status.items():
        final_stage = status['final_stage']
        
        if final_stage == num_stages:
            marker = "★ CHAMPION"
        elif final_stage == 0:
            marker = "(failed stage 1)"
        else:
            marker = ""
        
        print(f"  {policy_name}: Stage {final_stage}/{num_stages} {marker}")
    
    print(f"{'='*width}")
    print(f"  Output directory: {output_dir}")
    print(f"  Results CSV:      {os.path.join(output_dir, 'results.csv')}")
    print(f"{'='*width}\n")


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse of quaternion (xyzw format)."""
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


# Terrain generation for MuJoCo heightfield
def generate_flat_terrain(nrow: int, ncol: int) -> np.ndarray:
    """Generate flat terrain (all zeros)."""
    return np.zeros((nrow, ncol), dtype=np.float32)


def generate_slope_terrain(nrow: int, ncol: int, slope: float, size_x: float) -> np.ndarray:
    """
    Generate sloped terrain.
    
    Args:
        nrow, ncol: Heightfield dimensions
        slope: Slope gradient (tan of angle, e.g., 0.087 for ~5 degrees)
        size_x: Physical size in x direction (meters)
    """
    x = np.linspace(-size_x, size_x, ncol)
    heights = slope * x
    heights = heights - heights.min()
    terrain = np.tile(heights, (nrow, 1)).astype(np.float32)
    return terrain


def generate_rough_terrain(nrow: int, ncol: int, random_height: float, seed: int = 42) -> np.ndarray:
    """
    Generate rough/random terrain.
    
    Args:
        nrow, ncol: Heightfield dimensions
        random_height: Maximum height variation (meters)
        seed: Random seed for reproducibility
    """
    rng = np.random.RandomState(seed)
    terrain = rng.uniform(0, random_height, (nrow, ncol)).astype(np.float32)
    return terrain


def set_heightfield_data(mj_model, terrain_data: np.ndarray):
    """
    Set MuJoCo heightfield data.
    
    Args:
        mj_model: MuJoCo model
        terrain_data: 2D array of height values (nrow x ncol), values in [0, 1] range
    """
    hfield_size = mj_model.hfield_size[0]
    max_height = hfield_size[2]
    
    normalized = terrain_data / max_height if max_height > 0 else terrain_data
    normalized = np.clip(normalized, 0, 1)
    
    mj_model.hfield_data[:] = normalized.flatten()


def setup_terrain(mj_model, terrain_cfg: dict, seed: int = 42):
    """
    Setup terrain based on configuration.
    
    Args:
        mj_model: MuJoCo model with heightfield
        terrain_cfg: Terrain config dict with 'type' and optional params
        seed: Random seed for rough terrain
    """
    nrow = mj_model.hfield_nrow[0]
    ncol = mj_model.hfield_ncol[0]
    size_x = mj_model.hfield_size[0][0]
    
    terrain_type = terrain_cfg['type']
    
    if terrain_type == 'plane':
        terrain_data = generate_flat_terrain(nrow, ncol)
    elif terrain_type == 'trimesh':
        if 'slope' in terrain_cfg and terrain_cfg.get('proportions', [0, 0, 0, 0])[1] > 0:
            terrain_data = generate_slope_terrain(nrow, ncol, terrain_cfg['slope'], size_x)
        elif 'random_height' in terrain_cfg:
            terrain_data = generate_rough_terrain(nrow, ncol, terrain_cfg['random_height'], seed)
        else:
            terrain_data = generate_flat_terrain(nrow, ncol)
    else:
        print(f"[!] Unknown terrain type '{terrain_type}', using flat")
        terrain_data = generate_flat_terrain(nrow, ncol)
    
    set_heightfield_data(mj_model, terrain_data)