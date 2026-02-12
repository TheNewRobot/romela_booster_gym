#!/usr/bin/env python3
"""
System Identification for Joint Dynamics

Optimizes per-joint simulation parameters (damping, armature, frictionloss)
to match real robot data using Nelder-Mead optimization.

Usage:
    python optimize_joint_dynamics.py --experiment hanging_test_01
    python optimize_joint_dynamics.py --experiment hanging_test_01 --joint 15
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.joint_data_utils import load_all_joint_data, JointData
from utils.mujoco_utils import (
    load_mujoco_model, get_joint_dof_indices,
    simulate_joint_with_params, LEG_JOINT_NAMES
)


# Physical bounds in log-space
# damping:      [0.1, 10.0]  → log: [-2.3, 2.3]
# armature:     [0.001, 1.0] → log: [-6.9, 0.0]
# frictionloss: [0.001, 2.0] → log: [-6.9, 0.7]
BOUNDS_LOG = np.array([
    [-2.3, 2.3],   # damping
    [-6.9, 0.0],   # armature
    [-6.9, 0.7],   # frictionloss
])


def optimize_single_joint(
    mj_model,
    joint_data: JointData,
    dof_idx: int,
    initial_params: Dict[str, float],
    subsample: int = 5,
    maxfev: int = 200,
) -> Dict[str, float]:
    """
    Optimize damping/armature/frictionloss for a single joint.

    Uses Nelder-Mead (gradient-free simplex) with soft bounds enforced
    via a quadratic penalty to keep parameters physically plausible.
    """
    timestamps = joint_data.timestamps[::subsample]
    cmd = joint_data.cmd_positions[::subsample]
    real = joint_data.actual_positions[::subsample]

    # Log-scale params for positivity
    x0 = np.array([
        np.log(max(initial_params['damping'], 1e-6)),
        np.log(max(initial_params['armature'], 1e-6)),
        np.log(max(initial_params['frictionloss'], 1e-6)),
    ], dtype=np.float64)
    x0 = np.clip(x0, BOUNDS_LOG[:, 0], BOUNDS_LOG[:, 1])

    eval_count = [0]

    def objective(log_params):
        # Soft bounds: heavy penalty outside physical range
        penalty = 0.0
        for i in range(3):
            if log_params[i] < BOUNDS_LOG[i, 0]:
                penalty += 1e3 * (BOUNDS_LOG[i, 0] - log_params[i]) ** 2
            elif log_params[i] > BOUNDS_LOG[i, 1]:
                penalty += 1e3 * (log_params[i] - BOUNDS_LOG[i, 1]) ** 2

        damping, armature, friction = np.exp(np.clip(
            log_params, BOUNDS_LOG[:, 0], BOUNDS_LOG[:, 1]
        ))
        sim_pos, _ = simulate_joint_with_params(
            mj_model, dof_idx, cmd,
            joint_data.kp, joint_data.kd,
            damping, armature, friction,
            initial_pos=real[0], timestamps=timestamps
        )
        loss = float(np.mean((sim_pos - real) ** 2))
        eval_count[0] += 1
        if eval_count[0] % 10 == 0:
            print(f"    eval {eval_count[0]}: loss={loss:.8f}, "
                  f"d={damping:.4f}, a={armature:.6f}, f={friction:.4f}")
        return loss + penalty

    print(f"    Initial loss: {objective(x0):.8f}")
    print(f"    Bounds: damping=[0.1, 10.0], armature=[0.001, 1.0], frictionloss=[0.001, 2.0]")

    result = minimize(
        objective, x0,
        method='Nelder-Mead',
        options={
            'maxfev': maxfev,
            'xatol': 1e-3,   # convergence tolerance in log-space
            'fatol': 1e-9,    # convergence tolerance in loss
            'adaptive': True, # scale simplex to parameter dimensionality
        },
    )

    best = np.exp(np.clip(result.x, BOUNDS_LOG[:, 0], BOUNDS_LOG[:, 1]))
    print(f"    Converged: {result.success}, evals: {result.nfev}, "
          f"final loss: {result.fun:.8f}")

    return {
        'damping': float(best[0]),
        'armature': float(best[1]),
        'frictionloss': float(best[2]),
        'final_loss': float(result.fun),
    }


def save_results(results: Dict[int, Dict], sim_params_path: str, output_path: str = None):
    """Save optimized parameters to sim_params.yaml."""
    with open(sim_params_path, 'r') as f:
        sim_params = yaml.safe_load(f)
    
    # Ensure structure exists
    sim_params.setdefault('mujoco', {}).setdefault('joint', {})
    
    # Merge per-joint dicts (preserve existing entries for joints not optimized)
    for key in ('damping', 'armature', 'frictionloss'):
        existing = sim_params['mujoco']['joint'].get(key, {})
        if not isinstance(existing, dict):
            existing = {}
        existing.update({j: round(r[key], 6) for j, r in results.items()})
        sim_params['mujoco']['joint'][key] = existing
    sim_params['mujoco']['joint']['_optimization_info'] = {
        'timestamp': datetime.now().isoformat(),
        'joints_optimized': list(results.keys()),
    }
    
    out = output_path or sim_params_path
    with open(out, 'w') as f:
        yaml.dump(sim_params, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved to {out}")


def plot_results(results: Dict[int, Dict], output_dir: Path):
    """Create summary bar plots."""
    output_dir.mkdir(exist_ok=True)
    
    joints = sorted(results.keys())
    names = [LEG_JOINT_NAMES.get(j, f"J{j}") for j in joints]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Optimization Results', fontsize=14, fontweight='bold')
    
    metrics = ['damping', 'armature', 'frictionloss', 'final_loss']
    colors = ['steelblue', 'coral', 'seagreen', 'purple']
    titles = ['Damping', 'Armature', 'Frictionloss', 'Final Loss (MSE)']
    
    for ax, metric, color, title in zip(axes.flat, metrics, colors, titles):
        vals = [results[j][metric] for j in joints]
        ax.bar(range(len(joints)), vals, color=color)
        ax.set_xticks(range(len(joints)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimization_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_dir / 'optimization_results.png'}")


def main():
    parser = argparse.ArgumentParser(description='Optimize joint dynamics parameters')
    parser.add_argument('--experiment', required=True, help='Experiment name')
    parser.add_argument('--robot-config', default='deploy/configs/T1.yaml')
    parser.add_argument('--mujoco-xml', default='resources/T1/T1_locomotion.xml')
    parser.add_argument('--sim-params', default='tests/sim2real/config/sim_params.yaml')
    parser.add_argument('--output', default=None, help='Output sim_params path')
    parser.add_argument('--joint', type=int, default=None, help='Optimize a single joint index (e.g. 15)')
    parser.add_argument('--maxfev', type=int, default=200, help='Max function evaluations per joint')
    parser.add_argument('--subsample', type=int, default=5, help='Subsample factor for data (default 5)')
    args = parser.parse_args()
    
    exp_dir = Path('tests/sim2real/data') / args.experiment
    
    # Validate paths
    for p in [exp_dir, args.robot_config, args.mujoco_xml, args.sim_params]:
        if not Path(p).exists():
            print(f"Error: {p} not found")
            return
    
    print(f"\n{'='*60}")
    print(f"Joint Dynamics Optimization")
    print(f"{'='*60}")
    print(f"Experiment: {args.experiment}")
    print(f"Max evals: {args.maxfev}, Subsample: {args.subsample}")
    
    # Load configs
    with open(args.robot_config) as f:
        robot_cfg = yaml.safe_load(f)
    with open(args.sim_params) as f:
        sim_params = yaml.safe_load(f)
    
    # Load model and data
    print(f"\nLoading MuJoCo model...")
    mj_model = load_mujoco_model(args.mujoco_xml)
    joint_to_dof = get_joint_dof_indices(mj_model)
    print(f"Joint→DOF mapping: {joint_to_dof}")
    
    print(f"\nLoading joint data...")
    joint_data = load_all_joint_data(exp_dir, robot_cfg)
    
    if not joint_data:
        print("No joint data found!")
        return
    
    # Initial params
    mj_joint = sim_params.get('mujoco', {}).get('joint', {})
    initial = {
        'damping': mj_joint.get('damping', 0.5) if not isinstance(mj_joint.get('damping'), dict) else 0.5,
        'armature': mj_joint.get('armature', 0.01) if not isinstance(mj_joint.get('armature'), dict) else 0.01,
        'frictionloss': mj_joint.get('frictionloss', 0.01) if not isinstance(mj_joint.get('frictionloss'), dict) else 0.01,
    }
    print(f"Initial params: {initial}")
    
    # Optimize each joint
    results = {}
    start = time.time()
    
    for jidx, jdata in sorted(joint_data.items()):
        if args.joint is not None and jidx != args.joint:
            continue
        if jidx not in joint_to_dof:
            print(f"\nSkipping joint {jidx}: not in MuJoCo model")
            continue
        
        print(f"\n{'='*50}")
        print(f"Optimizing joint {jidx}: {jdata.joint_name}")
        print(f"{'='*50}")
        
        results[jidx] = optimize_single_joint(
            mj_model, jdata, joint_to_dof[jidx],
            initial, subsample=args.subsample, maxfev=args.maxfev
        )
        
        r = results[jidx]
        print(f"  → damping={r['damping']:.4f}, armature={r['armature']:.6f}, "
              f"friction={r['frictionloss']:.4f}, loss={r['final_loss']:.6f}")
    
    elapsed = time.time() - start
    
    # Save and plot
    save_results(results, args.sim_params, args.output)
    plot_results(results, exp_dir / 'plots')
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Complete! Time: {elapsed:.1f}s, Joints: {len(results)}")
    print(f"{'='*60}")
    print(f"{'Joint':<6} {'Name':<20} {'Damp':<8} {'Arma':<10} {'Fric':<8} {'Loss':<10}")
    print('-'*62)
    for j in sorted(results):
        r = results[j]
        print(f"{j:<6} {LEG_JOINT_NAMES.get(j,'?'):<20} {r['damping']:<8.4f} "
              f"{r['armature']:<10.6f} {r['frictionloss']:<8.4f} {r['final_loss']:<10.6f}")


if __name__ == '__main__':
    main()