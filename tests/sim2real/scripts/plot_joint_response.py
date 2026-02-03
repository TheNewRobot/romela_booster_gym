#!/usr/bin/env python3
"""
Plot Joint Response Data

Visualizes step and sinusoid responses from sysid joint dynamics tests.
Creates diagnostic plots to verify data quality before sim2real comparison.

Usage:
    python plot_joint_response.py --experiment hanging_test_01
    python plot_joint_response.py --data-dir tests/sim2real/data/hanging_test_01
"""

import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_csv_header(filepath):
    """Extract metadata from CSV header comments."""
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                key_value = line[1:].strip().split(':', 1)
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    metadata[key] = value
            else:
                break
    return metadata


def load_joint_data(filepath):
    """Load joint data CSV, skipping metadata header."""
    df = pd.read_csv(filepath, comment='#')
    metadata = parse_csv_header(filepath)
    return df, metadata


def plot_joint_response(df, metadata, output_path):
    """Create diagnostic plot for a single joint."""
    joint_name = metadata.get('joint_name', 'Unknown')
    joint_idx = metadata.get('joint_index', '?')
    experiment = metadata.get('experiment', 'unknown')
    
    # Normalize timestamps to start at 0
    df = df.copy()
    df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    
    # Separate step and sine data
    step_data = df[df['test_type'] == 'step']
    sine_data = df[df['test_type'] == 'sine']
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"Joint {joint_idx}: {joint_name}\nExperiment: {experiment}", fontsize=14, fontweight='bold')
    
    # === Step Response Plot ===
    ax1 = axes[0]
    if len(step_data) > 0:
        amplitudes = step_data['test_param'].unique()
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(amplitudes)))
        
        for amp, color in zip(sorted(amplitudes), colors):
            mask = step_data['test_param'] == amp
            subset = step_data[mask]
            # Reset time for each test segment
            t = subset['time'].values - subset['time'].values[0]
            ax1.plot(t, subset['cmd_position'], '--', color=color, alpha=0.7, linewidth=1.5)
            ax1.plot(t, subset['actual_position'], '-', color=color, label=f'{amp:+.3f} rad', linewidth=1.5)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (rad)')
        ax1.set_title('Step Responses (dashed=cmd, solid=actual)')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Add tracking error info
        errors = []
        for amp in amplitudes:
            mask = step_data['test_param'] == amp
            subset = step_data[mask]
            # Use last 20% of data for steady-state error
            n_ss = max(1, len(subset) // 5)
            ss_error = np.mean(np.abs(subset['cmd_position'].iloc[-n_ss:] - subset['actual_position'].iloc[-n_ss:]))
            errors.append(ss_error)
        avg_ss_error = np.mean(errors)
        ax1.text(0.02, 0.98, f'Avg steady-state error: {avg_ss_error:.4f} rad', 
                 transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax1.text(0.5, 0.5, 'No step response data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Step Responses (no data)')
    
    # === Sinusoid Response Plot ===
    ax2 = axes[1]
    if len(sine_data) > 0:
        frequencies = sine_data['test_param'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))
        
        for freq, color in zip(sorted(frequencies), colors):
            mask = sine_data['test_param'] == freq
            subset = sine_data[mask]
            # Reset time for each test segment
            t = subset['time'].values - subset['time'].values[0]
            ax2.plot(t, subset['cmd_position'], '--', color=color, alpha=0.7, linewidth=1.5)
            ax2.plot(t, subset['actual_position'], '-', color=color, label=f'{freq} Hz', linewidth=1.5)
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position (rad)')
        ax2.set_title('Sinusoid Responses (dashed=cmd, solid=actual)')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Add phase lag / tracking info
        rms_errors = []
        for freq in frequencies:
            mask = sine_data['test_param'] == freq
            subset = sine_data[mask]
            rms_error = np.sqrt(np.mean((subset['cmd_position'] - subset['actual_position'])**2))
            rms_errors.append(rms_error)
        avg_rms_error = np.mean(rms_errors)
        ax2.text(0.02, 0.98, f'Avg RMS tracking error: {avg_rms_error:.4f} rad', 
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, 'No sinusoid data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Sinusoid Responses (no data)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'joint_idx': joint_idx,
        'joint_name': joint_name,
        'n_step_records': len(step_data),
        'n_sine_records': len(sine_data),
        'output_path': output_path
    }


def main():
    parser = argparse.ArgumentParser(description='Plot joint response data from sysid tests')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment name (looks in tests/sim2real/data/{experiment})')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Direct path to data directory (overrides --experiment)')
    args = parser.parse_args()
    
    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.experiment:
        data_dir = Path('tests/sim2real/data') / args.experiment
    else:
        print("Error: Provide either --experiment or --data-dir")
        return
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Create plots directory
    plots_dir = data_dir / 'plots' / 'raw'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all joint CSV files
    csv_pattern = str(data_dir / 'joint_*.csv')
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        print(f"No joint CSV files found in {data_dir}")
        print(f"Pattern searched: {csv_pattern}")
        return
    
    print(f"\n{'='*60}")
    print(f"Joint Response Diagnostic Plots")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Found {len(csv_files)} joint CSV files")
    print(f"Output directory: {plots_dir}")
    print(f"{'='*60}\n")
    
    # Process each CSV
    results = []
    for csv_file in csv_files:
        print(f"Processing: {os.path.basename(csv_file)}")
        try:
            df, metadata = load_joint_data(csv_file)
            
            joint_idx = metadata.get('joint_index', 'unknown')
            joint_name = metadata.get('joint_name', 'Unknown')
            output_filename = f"joint_{joint_idx}_{joint_name}.png"
            output_path = plots_dir / output_filename
            
            result = plot_joint_response(df, metadata, output_path)
            results.append(result)
            print(f"  -> Saved: {output_filename}")
            print(f"     Step records: {result['n_step_records']}, Sine records: {result['n_sine_records']}")
        except Exception as e:
            print(f"  -> Error: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Joint':<8} {'Name':<20} {'Step':<10} {'Sine':<10}")
    print(f"{'-'*8} {'-'*20} {'-'*10} {'-'*10}")
    for r in results:
        print(f"{r['joint_idx']:<8} {r['joint_name']:<20} {r['n_step_records']:<10} {r['n_sine_records']:<10}")
    print(f"\nPlots saved to: {plots_dir}")
    print(f"\nUpload the PNG files for joints 15, 16, 21, 22 (and optionally 11 or 14 as baseline)")


if __name__ == '__main__':
    main()