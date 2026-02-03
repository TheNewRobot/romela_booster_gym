#!/usr/bin/env python3
"""
Differentiable System Identification for Joint Dynamics

Optimizes per-joint simulation parameters (damping, armature, frictionloss)
to match real robot data using gradient descent.

Usage:
    python optimize_joint_dynamics.py --experiment hanging_test_01
    python optimize_joint_dynamics.py --experiment hanging_test_01 --iterations 500 --lr 0.01
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

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.joint_data_utils import load_all_joint_data, JointData
from utils.mujoco_utils import (
    load_mujoco_model, get_joint_dof_indices, 
    simulate_joint_with_params, LEG_JOINT_NAMES
)


def optimize_single_joint(
    mj_model,
    joint_data: JointData,
    dof_idx: int,
    initial_params: Dict[str, float],
    n_iterations: int = 500,
    learning_rate: float = 0.01,
    subsample: int = 10,
) -> Dict[str, float]:
    """
    Optimize damping/armature/frictionloss for a single joint.
    
    Uses gradient descent with finite differences.
    """
    # Subsample for speed
    cmd = joint_data.cmd_positions[::subsample]
    real = joint_data.actual_positions[::subsample]
    
    # Log-scale params for positivity
    params = np.array([
        np.log(max(initial_params['damping'], 1e-6)),
        np.log(max(initial_params['armature'], 1e-6)),
        np.log(max(initial_params['frictionloss'], 1e-6)),
    ], dtype=np.float32)
    
    best_loss = float('inf')
    best_params = params.copy()
    
    eps = 1e-4
    momentum = np.zeros_like(params)
    beta = 0.9
    
    for it in range(n_iterations):
        # Current params
        damping, armature, friction = np.exp(params)
        
        # Simulate
        sim_pos, _ = simulate_joint_with_params(
            mj_model, dof_idx, cmd,
            joint_data.kp, joint_data.kd,
            damping, armature, friction,
            initial_pos=real[0]
        )
        
        # Loss
        loss = np.mean((sim_pos - real) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()
        
        # Gradients via finite differences
        grads = np.zeros(3)
        for i in range(3):
            p_plus = params.copy()
            p_plus[i] += eps
            
            sim_plus, _ = simulate_joint_with_params(
                mj_model, dof_idx, cmd,
                joint_data.kp, joint_data.kd,
                np.exp(p_plus[0]), np.exp(p_plus[1]), np.exp(p_plus[2]),
                initial_pos=real[0]
            )
            loss_plus = np.mean((sim_plus - real) ** 2)
            grads[i] = (loss_plus - loss) / eps
        
        # Update with momentum
        momentum = beta * momentum + (1 - beta) * grads
        params = params - learning_rate * momentum
        params = np.clip(params, -10, 5)
        
        if it % 100 == 0:
            print(f"    Iter {it}: loss={loss:.6f}, d={damping:.4f}, a={armature:.6f}, f={friction:.4f}")
    
    return {
        'damping': float(np.exp(best_params[0])),
        'armature': float(np.exp(best_params[1])),
        'frictionloss': float(np.exp(best_params[2])),
        'final_loss': float(best_loss),
    }


def save_results(results: Dict[int, Dict], sim_params_path: str, output_path: str = None):
    """Save optimized parameters to sim_params.yaml."""
    with open(sim_params_path, 'r') as f:
        sim_params = yaml.safe_load(f)
    
    # Ensure structure exists
    sim_params.setdefault('mujoco', {}).setdefault('joint', {})
    
    # Build per-joint dicts
    sim_params['mujoco']['joint']['damping'] = {j: round(r['damping'], 6) for j, r in results.items()}
    sim_params['mujoco']['joint']['armature'] = {j: round(r['armature'], 6) for j, r in results.items()}
    sim_params['mujoco']['joint']['frictionloss'] = {j: round(r['frictionloss'], 6) for j, r in results.items()}
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
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
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
    print(f"Iterations: {args.iterations}, LR: {args.lr}")
    
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
        if jidx not in joint_to_dof:
            print(f"\nSkipping joint {jidx}: not in MuJoCo model")
            continue
        
        print(f"\n{'='*50}")
        print(f"Optimizing joint {jidx}: {jdata.joint_name}")
        print(f"{'='*50}")
        
        results[jidx] = optimize_single_joint(
            mj_model, jdata, joint_to_dof[jidx],
            initial, args.iterations, args.lr
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