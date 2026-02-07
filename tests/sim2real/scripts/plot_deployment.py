#!/usr/bin/env python3
"""
Deployment diagnostic dashboard.
Reads deployment_log.csv from deploy/data/<task>/<experiment>/ and generates
analysis plots + stats.txt in the same folder.

Usage (from repo root):
    python deploy/plot_deployment.py --data deploy/data/T1/deploy_log_obs_2026-02-08_03-58-08
    python deploy/plot_deployment.py --data deploy/data/T1/deploy_log_obs_2026-02-08_03-58-08 --show
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Joint mapping (indices into 23-joint arrays)
LEFT_LEG_INDICES = [11, 12, 13, 14, 15, 16]
RIGHT_LEG_INDICES = [17, 18, 19, 20, 21, 22]
LEG_JOINT_NAMES = ["Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll"]

CMD_TRIM_THRESHOLD = 0.01


def load_data(folder):
    path = os.path.join(folder, "deployment_log.csv")
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return None
    df = pd.read_csv(path, comment="#")
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def auto_trim(df):
    """Find time range where commands are active (magnitude > threshold)."""
    mag = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2 + df["vyaw"] ** 2)
    active = mag > CMD_TRIM_THRESHOLD
    if not active.any():
        return df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
    first = df["timestamp"][active].iloc[0]
    last = df["timestamp"][active].iloc[-1]
    margin = 0.5
    return max(first - margin, df["timestamp"].iloc[0]), last + margin


def trim_df(df, t_start, t_end):
    if df is None:
        return None
    mask = (df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)
    return df[mask].reset_index(drop=True)


def plot_leg(df, leg_indices, leg_name, output_path, t_start, show=False):
    """6 rows x 2 cols: left col = tracking (cmd vs actual), right col = torque (tau_est)."""
    fig, axes = plt.subplots(6, 2, figsize=(14, 14), sharex=True)
    fig.suptitle(f"{leg_name} Leg — Joint Tracking & Torques", fontsize=14, y=0.99)

    t = df["timestamp"] - t_start

    for row, (joint_idx, joint_name) in enumerate(zip(leg_indices, LEG_JOINT_NAMES)):
        # Tracking panel
        ax_track = axes[row, 0]
        q_cmd_col = f"q_cmd_{joint_idx}"
        q_act_col = f"q_act_{joint_idx}"
        if q_cmd_col in df.columns:
            ax_track.plot(t, df[q_cmd_col], color="C0", alpha=0.8, linewidth=1.0, label="cmd")
        if q_act_col in df.columns:
            ax_track.plot(t, df[q_act_col], color="C1", alpha=0.8, linewidth=1.0, label="actual")
        ax_track.grid(True, alpha=0.3, linewidth=0.5)
        ax_track.set_ylabel(f"{joint_name}\n(rad)", fontsize=8)
        if row == 0:
            ax_track.legend(fontsize=7, loc="upper right")

        # Torque panel
        ax_tau = axes[row, 1]
        tau_col = f"tau_est_{joint_idx}"
        if tau_col in df.columns:
            ax_tau.plot(t, df[tau_col], color="C2", alpha=0.8, linewidth=1.0)
        ax_tau.set_ylabel("τ (Nm)", fontsize=8)
        ax_tau.grid(True, alpha=0.3, linewidth=0.5)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[0, 0].set_title("Position Tracking (cmd vs actual)", fontsize=10)
    axes[0, 1].set_title("Estimated Torque (τ_est)", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_overview(df, t_start, output_path, show=False):
    """Commands timeline, vyaw vs gyro_z, IMU orientation, IMU gyro."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Deployment Overview", fontsize=14, y=0.98)

    t = df["timestamp"] - t_start

    # Panel 1: Command timeline
    ax = axes[0]
    ax.plot(t, df["vx"], label="vx", linewidth=1.2)
    ax.plot(t, df["vy"], label="vy", linewidth=1.2)
    ax.plot(t, df["vyaw"], label="vyaw", linewidth=1.2)
    ax.set_ylabel("Command")
    ax.set_title("Velocity Commands", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Panel 2: Vyaw vs gyro_z
    ax = axes[1]
    ax.plot(t, df["vyaw"], label="vyaw (cmd)", linewidth=1.2)
    ax.plot(t, df["gyro_z"], label="gyro_z (actual)", alpha=0.8, linewidth=1.0)
    ax.set_ylabel("rad/s")
    ax.set_title("Yaw Rate: Commanded vs Actual", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Panel 3: IMU orientation
    ax = axes[2]
    ax.plot(t, df["roll"], label="roll", linewidth=1.2)
    ax.plot(t, df["pitch"], label="pitch", linewidth=1.2)
    ax.plot(t, df["yaw"], label="yaw", linewidth=1.2)
    ax.set_ylabel("rad")
    ax.set_title("IMU Orientation (RPY)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Panel 4: IMU angular velocity
    ax = axes[3]
    ax.plot(t, df["gyro_x"], label="gyro_x", linewidth=1.2)
    ax.plot(t, df["gyro_y"], label="gyro_y", linewidth=1.2)
    ax.plot(t, df["gyro_z"], label="gyro_z", linewidth=1.2)
    ax.set_ylabel("rad/s")
    ax.set_xlabel("Time (s)")
    ax.set_title("IMU Angular Velocity", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_temperature(df, t_start, output_path, show=False):
    """6 rows x 2 cols: left col = left leg temps, right col = right leg temps."""
    fig, axes = plt.subplots(6, 2, figsize=(14, 14), sharex=True)
    fig.suptitle("Motor Temperature", fontsize=14, y=0.99)


    t = df["timestamp"] - t_start

    for row, joint_name in enumerate(LEG_JOINT_NAMES):
        # Left leg
        ax_left = axes[row, 0]
        left_idx = LEFT_LEG_INDICES[row]
        temp_col = f"temp_{left_idx}"
        if temp_col in df.columns:
            vals = df[temp_col]
            ax_left.plot(t, vals, color="C3", linewidth=1.2)
            peak = vals.max()
            ax_left.axhline(y=peak, color="C3", linestyle="--", alpha=0.4, linewidth=0.5)
            ax_left.text(t.iloc[-1], peak, f" {peak:.1f}", fontsize=7,
                         va="bottom", color="C3")
        ax_left.set_ylabel(f"{joint_name}\n(°C)", fontsize=8)
        ax_left.grid(True, alpha=0.3, linewidth=0.5)

        # Right leg
        ax_right = axes[row, 1]
        right_idx = RIGHT_LEG_INDICES[row]
        temp_col = f"temp_{right_idx}"
        if temp_col in df.columns:
            vals = df[temp_col]
            ax_right.plot(t, vals, color="C4", linewidth=1.2)
            peak = vals.max()
            ax_right.axhline(y=peak, color="C4", linestyle="--", alpha=0.4, linewidth=0.5)
            ax_right.text(t.iloc[-1], peak, f" {peak:.1f}", fontsize=7,
                          va="bottom", color="C4")
        ax_right.set_ylabel(f"(°C)", fontsize=8)
        ax_right.grid(True, alpha=0.3, linewidth=0.5)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[0, 0].set_title("Left Leg Temperature", fontsize=10)
    axes[0, 1].set_title("Right Leg Temperature", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def compute_stats(df, t_start, t_end):
    lines = []
    duration = t_end - t_start
    lines.append(f"Duration: {duration:.1f}s")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Effective rate: {len(df) / duration:.1f} Hz")
    lines.append("")

    # Joint tracking RMS and peak error
    lines.append("Joint Tracking Error (rad):")
    lines.append(f"  {'Joint':<20s} {'RMS':>8s} {'Peak':>8s}")
    for indices, side in [(LEFT_LEG_INDICES, "L"), (RIGHT_LEG_INDICES, "R")]:
        for joint_idx, joint_name in zip(indices, LEG_JOINT_NAMES):
            q_act = f"q_act_{joint_idx}"
            q_cmd = f"q_cmd_{joint_idx}"
            if q_act not in df.columns or q_cmd not in df.columns:
                continue
            err = df[q_cmd].values - df[q_act].values
            rms = np.sqrt(np.mean(err ** 2))
            peak = np.max(np.abs(err))
            lines.append(f"  {side}_{joint_name:<18s} {rms:8.4f} {peak:8.4f}")
    lines.append("")

    # Joint torque stats
    lines.append("Joint Torque τ_est (Nm):")
    lines.append(f"  {'Joint':<20s} {'RMS':>8s} {'Peak':>8s}")
    for indices, side in [(LEFT_LEG_INDICES, "L"), (RIGHT_LEG_INDICES, "R")]:
        for joint_idx, joint_name in zip(indices, LEG_JOINT_NAMES):
            tau_col = f"tau_est_{joint_idx}"
            if tau_col not in df.columns:
                continue
            tau = df[tau_col].values
            rms = np.sqrt(np.mean(tau ** 2))
            peak = np.max(np.abs(tau))
            lines.append(f"  {side}_{joint_name:<18s} {rms:8.3f} {peak:8.3f}")
    lines.append("")

    # Motor temperature
    lines.append("Motor Temperature (°C):")
    lines.append(f"  {'Joint':<20s} {'Start':>8s} {'Peak':>8s} {'Final':>8s}")
    for indices, side in [(LEFT_LEG_INDICES, "L"), (RIGHT_LEG_INDICES, "R")]:
        for joint_idx, joint_name in zip(indices, LEG_JOINT_NAMES):
            temp_col = f"temp_{joint_idx}"
            if temp_col not in df.columns:
                continue
            vals = df[temp_col].values
            lines.append(f"  {side}_{joint_name:<18s} {vals[0]:8.1f} {vals.max():8.1f} {vals[-1]:8.1f}")
    lines.append("")

    # Action smoothness
    act_cols = [c for c in df.columns if c.startswith("action_")]
    if act_cols:
        actions = df[act_cols].values
        deltas = np.diff(actions, axis=0)
        smoothness = np.mean(np.abs(deltas))
        lines.append(f"Action Smoothness (mean |Δaction|): {smoothness:.4f}")
        lines.append("")

    # Yaw rate tracking
    if "vyaw" in df.columns and "gyro_z" in df.columns:
        yaw_err = df["vyaw"].values - df["gyro_z"].values
        yaw_rms = np.sqrt(np.mean(yaw_err ** 2))
        lines.append(f"Yaw Rate Tracking RMS: {yaw_rms:.4f} rad/s")
        lines.append("")

    return lines


def main():
    parser = argparse.ArgumentParser(description="Deployment diagnostic dashboard")
    parser.add_argument("--data", required=True, help="Path to experiment folder")
    parser.add_argument("--show", action="store_true", help="Show interactive plots")
    parser.add_argument("--no-trim", action="store_true", help="Skip auto-trimming")
    args = parser.parse_args()

    folder = args.data
    print(f"Loading data from: {folder}")

    df = load_data(folder)
    if df is None:
        return

    # Auto-trim
    if not args.no_trim:
        t_start, t_end = auto_trim(df)
        print(f"Auto-trimmed to [{t_start:.2f}, {t_end:.2f}]s")
        df = trim_df(df, t_start, t_end)
    else:
        t_start = df["timestamp"].iloc[0]
        t_end = df["timestamp"].iloc[-1]

    print("Generating plots...")

    plot_leg(df, LEFT_LEG_INDICES, "Left",
             os.path.join(folder, "tracking_left_leg.png"), t_start, args.show)

    plot_leg(df, RIGHT_LEG_INDICES, "Right",
             os.path.join(folder, "tracking_right_leg.png"), t_start, args.show)

    plot_overview(df, t_start,
                  os.path.join(folder, "overview.png"), args.show)

    plot_temperature(df, t_start,
                     os.path.join(folder, "temperature.png"), args.show)

    # Stats
    print("Computing stats...")
    stats_lines = compute_stats(df, t_start, t_end)

    header = f"Deployment: {os.path.basename(folder)}"
    stats_text = header + "\n" + "\n".join(stats_lines)

    stats_path = os.path.join(folder, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"  Saved {stats_path}")

    print("\n" + stats_text)
    print("\nDone.")


if __name__ == "__main__":
    main()