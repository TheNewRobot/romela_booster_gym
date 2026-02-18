"""Plotting functions for deployment and Vicon data visualization."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def plot_deployment_mocap(
    times: np.ndarray,
    vicon_base_pos: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vyaw: np.ndarray,
    output_dir: Path,
):
    """Deployment + mocap plots: XY path and time-series.

    Saves two files:
        - xy_path.png: top-down XY trajectory
        - timeseries.png: height, distance, commands (shared time axis)

    Expects positions already transformed to the robot's body frame
    (start at origin, initial heading = +X) via transform_positions_to_body_frame.
    """
    # XY top-down path
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(vicon_base_pos[:, 0], vicon_base_pos[:, 1], '-', color='blue',
            linewidth=1.5)
    ax.plot(0, 0, 'ko', markersize=8, label='Start')
    ax.plot(vicon_base_pos[-1, 0], vicon_base_pos[-1, 1], 'rs',
            markersize=8, label='End')
    ax.set_xlabel('Forward (m)')
    ax.set_ylabel('Left (m)')
    ax.set_title('XY Path (Top-Down)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_dir / 'xy_path.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Stacked time-series
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(times, vicon_base_pos[:, 2], '-', color='blue', linewidth=1.2)
    axes[0].set_ylabel('\u0394Z (m)')
    axes[0].set_title('Base Height Change')
    axes[0].grid(True, alpha=0.3)

    dist = np.sqrt(vicon_base_pos[:, 0]**2 + vicon_base_pos[:, 1]**2)
    axes[1].plot(times, dist, '-', color='blue', linewidth=1.2)
    axes[1].set_ylabel('Distance (m)')
    axes[1].set_title('Distance from Start')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, vx, label='vx', linewidth=1.2)
    axes[2].plot(times, vy, label='vy', linewidth=1.2)
    axes[2].plot(times, vyaw, label='vyaw', linewidth=1.2)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Command')
    axes[2].set_title('Velocity Commands')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
