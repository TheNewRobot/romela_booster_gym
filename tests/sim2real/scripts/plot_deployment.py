#!/usr/bin/env python3
"""
Deployment diagnostic dashboard.
Reads CSVs from deploy/data/<task>/<timestamp>/ and generates
analysis plots + stats.txt in the same folder.

Usage:
    python tests/sim2real/scripts/plot_deployment.py --data deploy/data/T1/2026-02-04_15-30-22
    python tests/sim2real/scripts/plot_deployment.py --data deploy/data/T1/2026-02-04_15-30-22 --show
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


def load_csv(folder, filename):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        print(f"  Warning: {filename} not found")
        return None
    return pd.read_csv(path)


def auto_trim(commands):
    """Find time range where commands are active (magnitude > threshold)."""
    mag = np.sqrt(commands["vx"] ** 2 + commands["vy"] ** 2 + commands["vyaw"] ** 2)
    active = mag > CMD_TRIM_THRESHOLD
    if not active.any():
        return commands["timestamp"].iloc[0], commands["timestamp"].iloc[-1]
    first = commands["timestamp"][active].iloc[0]
    last = commands["timestamp"][active].iloc[-1]
    # Add small margin
    margin = 0.5
    return max(first - margin, commands["timestamp"].iloc[0]), last + margin


def trim_df(df, t_start, t_end):
    if df is None:
        return None
    mask = (df["timestamp"] >= t_start) & (df["timestamp"] <= t_end)
    return df[mask].reset_index(drop=True)


def plot_leg(actual, commanded, leg_indices, leg_name, output_path, t_start, show=False):
    """6 rows x 2 cols: left col = tracking (cmd vs actual), right col = torque (tau_est)."""
    fig, axes = plt.subplots(6, 2, figsize=(14, 16), sharex=True)
    fig.suptitle(f"{leg_name} Leg — Joint Tracking & Torques", fontsize=14, y=0.98)

    for row, (joint_idx, joint_name) in enumerate(zip(leg_indices, LEG_JOINT_NAMES)):
        q_cmd_col = f"q_{joint_idx}"
        q_act_col = f"q_{joint_idx}"
        tau_col = f"tau_{joint_idx}"

        # Tracking panel
        ax_track = axes[row, 0]
        if commanded is not None and q_cmd_col in commanded.columns:
            ax_track.plot(commanded["timestamp"] - t_start, commanded[q_cmd_col],
                          color="C0", alpha=0.7, linewidth=0.5, label="cmd")
        if actual is not None and q_act_col in actual.columns:
            ax_track.plot(actual["timestamp"] - t_start, actual[q_act_col],
                          color="C1", alpha=0.7, linewidth=0.5, label="actual")
        ax_track.set_ylabel(f"{joint_name}\n(rad)", fontsize=8)
        if row == 0:
            ax_track.legend(fontsize=7, loc="upper right")

        # Torque panel
        ax_tau = axes[row, 1]
        if actual is not None and tau_col in actual.columns:
            ax_tau.plot(actual["timestamp"] - t_start, actual[tau_col],
                        color="C2", alpha=0.7, linewidth=0.5)
        ax_tau.set_ylabel("τ (Nm)", fontsize=8)

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


def plot_overview(commands, imu, policy_actions, t_start, output_path, show=False):
    """Commands timeline, vyaw vs gyro_z, IMU orientation, IMU gyro."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Deployment Overview", fontsize=14, y=0.98)

    # Panel 1: Command timeline
    ax = axes[0]
    if commands is not None:
        t = commands["timestamp"] - t_start
        ax.plot(t, commands["vx"], label="vx", linewidth=0.8)
        ax.plot(t, commands["vy"], label="vy", linewidth=0.8)
        ax.plot(t, commands["vyaw"], label="vyaw", linewidth=0.8)
    ax.set_ylabel("Command")
    ax.set_title("Velocity Commands", fontsize=10)
    ax.legend(fontsize=8)

    # Panel 2: Vyaw vs gyro_z
    ax = axes[1]
    if commands is not None:
        ax.plot(commands["timestamp"] - t_start, commands["vyaw"],
                label="vyaw (cmd)", linewidth=0.8)
    if imu is not None:
        ax.plot(imu["timestamp"] - t_start, imu["gyro_z"],
                label="gyro_z (actual)", alpha=0.7, linewidth=0.5)
    ax.set_ylabel("rad/s")
    ax.set_title("Yaw Rate: Commanded vs Actual", fontsize=10)
    ax.legend(fontsize=8)

    # Panel 3: IMU orientation
    ax = axes[2]
    if imu is not None:
        t = imu["timestamp"] - t_start
        ax.plot(t, imu["roll"], label="roll", linewidth=0.8)
        ax.plot(t, imu["pitch"], label="pitch", linewidth=0.8)
        ax.plot(t, imu["yaw"], label="yaw", linewidth=0.8)
    ax.set_ylabel("rad")
    ax.set_title("IMU Orientation (RPY)", fontsize=10)
    ax.legend(fontsize=8)

    # Panel 4: IMU angular velocity
    ax = axes[3]
    if imu is not None:
        t = imu["timestamp"] - t_start
        ax.plot(t, imu["gyro_x"], label="gyro_x", linewidth=0.8)
        ax.plot(t, imu["gyro_y"], label="gyro_y", linewidth=0.8)
        ax.plot(t, imu["gyro_z"], label="gyro_z", linewidth=0.8)
    ax.set_ylabel("rad/s")
    ax.set_xlabel("Time (s)")
    ax.set_title("IMU Angular Velocity", fontsize=10)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Saved {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def compute_stats(commands, actual, commanded, imu, policy_actions, t_start, t_end):
    lines = []
    duration = t_end - t_start
    lines.append(f"Duration: {duration:.1f}s")
    lines.append("")

    # Joint tracking RMS and peak error
    if actual is not None and commanded is not None:
        # Resample commanded to actual timestamps via nearest-neighbor
        actual_t = actual["timestamp"].values
        cmd_t = commanded["timestamp"].values

        lines.append("Joint Tracking Error (rad):")
        lines.append(f"  {'Joint':<20s} {'RMS':>8s} {'Peak':>8s}")
        for indices, side in [(LEFT_LEG_INDICES, "L"), (RIGHT_LEG_INDICES, "R")]:
            for joint_idx, joint_name in zip(indices, LEG_JOINT_NAMES):
                q_col = f"q_{joint_idx}"
                if q_col not in actual.columns or q_col not in commanded.columns:
                    continue
                # Nearest-neighbor interpolation of commanded onto actual timestamps
                cmd_interp = np.interp(actual_t, cmd_t, commanded[q_col].values)
                err = cmd_interp - actual[q_col].values
                rms = np.sqrt(np.mean(err ** 2))
                peak = np.max(np.abs(err))
                lines.append(f"  {side}_{joint_name:<18s} {rms:8.4f} {peak:8.4f}")
        lines.append("")

    # Joint torque stats
    if actual is not None:
        lines.append("Joint Torque τ_est (Nm):")
        lines.append(f"  {'Joint':<20s} {'RMS':>8s} {'Peak':>8s}")
        for indices, side in [(LEFT_LEG_INDICES, "L"), (RIGHT_LEG_INDICES, "R")]:
            for joint_idx, joint_name in zip(indices, LEG_JOINT_NAMES):
                tau_col = f"tau_{joint_idx}"
                if tau_col not in actual.columns:
                    continue
                tau = actual[tau_col].values
                rms = np.sqrt(np.mean(tau ** 2))
                peak = np.max(np.abs(tau))
                lines.append(f"  {side}_{joint_name:<18s} {rms:8.3f} {peak:8.3f}")
        lines.append("")

    # Action smoothness
    if policy_actions is not None:
        act_cols = [c for c in policy_actions.columns if c.startswith("action_")]
        if act_cols:
            actions = policy_actions[act_cols].values
            deltas = np.diff(actions, axis=0)
            smoothness = np.mean(np.abs(deltas))
            lines.append(f"Action Smoothness (mean |Δaction|): {smoothness:.4f}")
            lines.append("")

    # Yaw rate tracking
    if commands is not None and imu is not None:
        cmd_vyaw = commands["vyaw"].values
        cmd_t = commands["timestamp"].values
        imu_t = imu["timestamp"].values
        gyro_z_interp = np.interp(cmd_t, imu_t, imu["gyro_z"].values)
        yaw_err = cmd_vyaw - gyro_z_interp
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

    # Load CSVs
    commands = load_csv(folder, "commands.csv")
    actual = load_csv(folder, "joint_states_actual.csv")
    commanded = load_csv(folder, "joint_states_commanded.csv")
    imu = load_csv(folder, "imu.csv")
    policy_obs = load_csv(folder, "policy_obs.csv")
    policy_actions = load_csv(folder, "policy_actions.csv")

    # Auto-trim
    if commands is not None and not args.no_trim:
        t_start, t_end = auto_trim(commands)
        print(f"Auto-trimmed to [{t_start:.2f}, {t_end:.2f}]s")
        commands = trim_df(commands, t_start, t_end)
        actual = trim_df(actual, t_start, t_end)
        commanded = trim_df(commanded, t_start, t_end)
        imu = trim_df(imu, t_start, t_end)
        policy_obs = trim_df(policy_obs, t_start, t_end)
        policy_actions = trim_df(policy_actions, t_start, t_end)
    else:
        t_start = 0.0
        t_end = commands["timestamp"].iloc[-1] if commands is not None else 0.0

    print("Generating plots...")

    # Left leg: tracking + torques
    plot_leg(actual, commanded, LEFT_LEG_INDICES, "Left",
             os.path.join(folder, "tracking_left_leg.png"), t_start, args.show)

    # Right leg: tracking + torques
    plot_leg(actual, commanded, RIGHT_LEG_INDICES, "Right",
             os.path.join(folder, "tracking_right_leg.png"), t_start, args.show)

    # Overview: commands, vyaw validation, IMU
    plot_overview(commands, imu, policy_actions, t_start,
                  os.path.join(folder, "overview.png"), args.show)

    # Stats
    print("Computing stats...")
    stats_lines = compute_stats(commands, actual, commanded, imu, policy_actions, t_start, t_end)

    header = f"Deployment: {os.path.basename(os.path.dirname(folder))}/{os.path.basename(folder)}"
    stats_text = header + "\n" + "\n".join(stats_lines)

    stats_path = os.path.join(folder, "stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"  Saved {stats_path}")

    print("\n" + stats_text)
    print("\nDone.")


if __name__ == "__main__":
    main()