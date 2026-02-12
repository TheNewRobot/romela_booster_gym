#!/usr/bin/env python3
"""
Compare Simulation vs Real Joint Responses

Loads real data and replays in MuJoCo with current sim_params.yaml,
then generates comparison plots and metrics to verify optimization.

Usage:
    python compare_sim_real.py --experiment hanging_test_01
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.joint_data_utils import load_all_joint_data, JointData
from utils.mujoco_utils import (
    load_mujoco_model, get_joint_dof_indices, apply_sim_params,
    simulate_joint_response, compute_metrics, LEG_JOINT_NAMES
)


def compare_joint(
    mj_model,
    joint_data: JointData,
    dof_idx: int,
    subsample: int = 5,
) -> Dict:
    """
    Compare sim vs real for a single joint.
    
    Returns dict with metrics and trajectories.
    """
    # Subsample for cleaner plots
    cmd = joint_data.cmd_positions[::subsample]
    real_pos = joint_data.actual_positions[::subsample]
    real_vel = joint_data.actual_velocities[::subsample]
    times = joint_data.timestamps[::subsample]

    # Simulate
    sim_pos, sim_vel = simulate_joint_response(
        mj_model, dof_idx, cmd,
        joint_data.kp, joint_data.kd,
        initial_pos=real_pos[0], timestamps=times
    )
    
    # Metrics
    metrics = compute_metrics(sim_pos, real_pos, sim_vel, real_vel)
    
    return {
        'times': times,
        'cmd': cmd,
        'real_pos': real_pos,
        'sim_pos': sim_pos,
        'real_vel': real_vel,
        'sim_vel': sim_vel,
        'metrics': metrics,
    }


def plot_comparison(
    joint_idx: int,
    joint_name: str,
    comparison: Dict,
    output_path: Path,
):
    """Create comparison plot for a single joint."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"Joint {joint_idx}: {joint_name} - Sim vs Real", fontsize=14, fontweight='bold')
    
    t = comparison['times']
    
    # Position
    ax = axes[0]
    ax.plot(t, comparison['cmd'], '--', color='gray', alpha=0.7, label='Command', linewidth=1)
    ax.plot(t, comparison['real_pos'], '-', color='blue', label='Real', linewidth=1.5)
    ax.plot(t, comparison['sim_pos'], '-', color='red', alpha=0.8, label='Sim', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (rad)')
    ax.set_title('Position Tracking')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add metrics text
    m = comparison['metrics']
    ax.text(0.02, 0.98, f"RMSE: {m['rmse_pos']:.4f} rad\nCorr: {m.get('correlation', 0):.3f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Velocity
    ax = axes[1]
    ax.plot(t, comparison['real_vel'], '-', color='blue', label='Real', linewidth=1.5)
    ax.plot(t, comparison['sim_vel'], '-', color='red', alpha=0.8, label='Sim', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (rad/s)')
    ax.set_title('Velocity')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if 'rmse_vel' in m:
        ax.text(0.02, 0.98, f"RMSE: {m['rmse_vel']:.4f} rad/s",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary(all_metrics: Dict[int, Dict], output_path: Path):
    """Create summary bar plot of all joints."""
    joints = sorted(all_metrics.keys())
    names = [LEG_JOINT_NAMES.get(j, f"J{j}") for j in joints]
    
    rmse = [all_metrics[j]['rmse_pos'] for j in joints]
    corr = [all_metrics[j].get('correlation', 0) for j in joints]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sim vs Real Comparison Summary', fontsize=14, fontweight='bold')
    
    # RMSE
    ax = axes[0]
    colors = ['green' if r < 0.02 else 'orange' if r < 0.05 else 'red' for r in rmse]
    ax.bar(range(len(joints)), rmse, color=colors)
    ax.set_xticks(range(len(joints)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('RMSE (rad)')
    ax.set_title('Position RMSE per Joint')
    ax.axhline(y=0.02, color='green', linestyle='--', alpha=0.5, label='Good (<0.02)')
    ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='OK (<0.05)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Correlation
    ax = axes[1]
    colors = ['green' if c > 0.95 else 'orange' if c > 0.8 else 'red' for c in corr]
    ax.bar(range(len(joints)), corr, color=colors)
    ax.set_xticks(range(len(joints)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Correlation')
    ax.set_title('Position Correlation per Joint')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare sim vs real joint responses')
    parser.add_argument('--experiment', required=True, help='Experiment name')
    parser.add_argument('--run-name', required=True, help='Name for this comparison run (e.g. default_params, optimized_all)')
    parser.add_argument('--robot-config', default='deploy/configs/T1.yaml')
    parser.add_argument('--mujoco-xml', default='resources/T1/T1_locomotion.xml')
    parser.add_argument('--sim-params', default='tests/sim2real/config/sim_params.yaml')
    args = parser.parse_args()

    exp_dir = Path('tests/sim2real/data') / args.experiment
    plots_dir = exp_dir / 'plots' / 'comparison' / args.run_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate
    for p in [exp_dir, args.robot_config, args.mujoco_xml, args.sim_params]:
        if not Path(p).exists():
            print(f"Error: {p} not found")
            return
    
    print(f"\n{'='*60}")
    print(f"Sim vs Real Comparison")
    print(f"{'='*60}")
    print(f"Experiment: {args.experiment}")
    print(f"Run: {args.run_name}")
    
    # Load configs
    with open(args.robot_config) as f:
        robot_cfg = yaml.safe_load(f)
    with open(args.sim_params) as f:
        sim_params = yaml.safe_load(f)
    
    # Load model and apply params
    print(f"\nLoading MuJoCo model and applying sim_params...")
    mj_model = load_mujoco_model(args.mujoco_xml)
    joint_to_dof = get_joint_dof_indices(mj_model)
    apply_sim_params(mj_model, sim_params, joint_to_dof)
    
    # Load data
    print(f"Loading joint data...")
    joint_data = load_all_joint_data(exp_dir, robot_cfg)
    
    if not joint_data:
        print("No joint data found!")
        return
    
    # Compare each joint
    all_metrics = {}
    print(f"\nComparing joints...")
    
    for jidx, jdata in sorted(joint_data.items()):
        if jidx not in joint_to_dof:
            print(f"  Skipping joint {jidx}: not in model")
            continue
        
        comparison = compare_joint(mj_model, jdata, joint_to_dof[jidx])
        all_metrics[jidx] = comparison['metrics']
        
        # Individual plot
        plot_path = plots_dir / f"comparison_joint_{jidx}_{jdata.joint_name}.png"
        plot_comparison(jidx, jdata.joint_name, comparison, plot_path)
        
        m = comparison['metrics']
        print(f"  Joint {jidx} ({jdata.joint_name}): RMSE={m['rmse_pos']:.4f}, Corr={m.get('correlation',0):.3f}")
    
    # Summary plot
    plot_summary(all_metrics, plots_dir / 'comparison_summary.png')

    # Save metrics CSV alongside the plots
    results_dir = exp_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([
        {'joint': j, 'name': LEG_JOINT_NAMES.get(j, f'J{j}'), **m}
        for j, m in sorted(all_metrics.items())
    ])
    metrics_path = results_dir / f'{args.run_name}.csv'
    metrics_df.to_csv(metrics_path, index=False)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Joint':<6} {'Name':<20} {'RMSE':<10} {'Correlation':<12}")
    print('-'*48)
    for j in sorted(all_metrics):
        m = all_metrics[j]
        print(f"{j:<6} {LEG_JOINT_NAMES.get(j,'?'):<20} {m['rmse_pos']:<10.4f} {m.get('correlation',0):<12.3f}")
    
    avg_rmse = np.mean([m['rmse_pos'] for m in all_metrics.values()])
    avg_corr = np.mean([m.get('correlation', 0) for m in all_metrics.values()])
    print('-'*48)
    print(f"{'AVG':<6} {'':<20} {avg_rmse:<10.4f} {avg_corr:<12.3f}")
    
    print(f"\nPlots saved to: {plots_dir}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == '__main__':
    main()