"""Replay .npz reference motions in MuJoCo for the Booster T1."""
import os
import sys
import time
import argparse
import numpy as np
import yaml

# Resolve default paths relative to the repo root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_T1_DIR = os.path.dirname(_SCRIPT_DIR)  # tests/integration/amp/T1
_REPO_ROOT = os.path.abspath(os.path.join(_T1_DIR, "..", "..", "..", ".."))
_DEFAULT_DATA_DIR = os.path.join(_T1_DIR, "data", "raw")
_DEFAULT_XML = os.path.join(_REPO_ROOT, "resources", "T1", "T1_locomotion.xml")
_CONFIG_PATH = os.path.join(_T1_DIR, "config", "config.yaml")

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("ERROR: mujoco not installed. pip install mujoco")
    sys.exit(1)

# Load config
with open(_CONFIG_PATH) as f:
    _CFG = yaml.safe_load(f)

JOINT_NAME_MAP = _CFG["joint_name_map"]
FREE_JOINT_NAMES = set(_CFG["free_joint_names"])
CAM_DISTANCE = _CFG["camera"]["distance"]
CAM_AZIMUTH = _CFG["camera"]["azimuth"]
CAM_ELEVATION = _CFG["camera"]["elevation"]

# Viewer state shared with key_callback
_viewer_state = {"camera_follow": True, "restart": False, "paused": False}


def _key_callback(keycode):
    """GLFW key callback — Space (32) pause, R (82) restart, O (79) camera toggle."""
    if keycode == 32:  # Space
        _viewer_state["paused"] = not _viewer_state["paused"]
        mode = "PAUSED" if _viewer_state["paused"] else "PLAYING"
        print(f"\r[{mode}]              ", end="", flush=True)
    elif keycode == 79:  # O
        _viewer_state["camera_follow"] = not _viewer_state["camera_follow"]
        mode = "FOLLOW" if _viewer_state["camera_follow"] else "FREE"
        print(f"\r[CAM: {mode}]          ", end="", flush=True)
    elif keycode == 82:  # R
        _viewer_state["restart"] = True
        _viewer_state["paused"] = False
        print(f"\r[RESTARTING...]        ", end="", flush=True)


def _update_camera(viewer, robot_pos):
    """Point the chase camera at the robot."""
    if _viewer_state["camera_follow"]:
        viewer.cam.lookat[:] = robot_pos
        viewer.cam.distance = CAM_DISTANCE
        viewer.cam.azimuth = CAM_AZIMUTH
        viewer.cam.elevation = CAM_ELEVATION


def get_mujoco_joint_names(model):
    """Extract joint names from MuJoCo model (skip the free joint)."""
    names = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if model.jnt_type[i] != 0:  # Skip free joints (type 0)
            names.append(name)
    return names


def build_joint_mapping(npz_joint_names, mj_joint_names):
    """
    Build index mapping from npz joint order to MuJoCo joint order.
    Returns: mapping array where mapping[mj_idx] = npz_idx, or None if positional.
    """
    if npz_joint_names is None:
        print("  No joint_names in npz → using positional ordering")
        return None
    
    # Clean up names
    if isinstance(npz_joint_names, np.ndarray):
        if npz_joint_names.ndim == 0:
            npz_joint_names = npz_joint_names.item()
        else:
            npz_joint_names = list(npz_joint_names)
    
    if isinstance(npz_joint_names, (list, tuple)):
        all_names = [str(n) for n in npz_joint_names]
    else:
        print(f"  joint_names type: {type(npz_joint_names)} → using positional ordering")
        return None

    # Filter out root/free joints — represented by qpos[0:7], not in qpos[7:]
    npz_names = [n for n in all_names if n not in FREE_JOINT_NAMES]

    print(f"  NPZ joints ({len(npz_names)}): {npz_names}")
    print(f"  MuJoCo joints ({len(mj_joint_names)}): {mj_joint_names}")

    # Try to build mapping
    mapping = []
    for mj_name in mj_joint_names:
        found = False
        for npz_idx, npz_name in enumerate(npz_names):
            # Direct match or through name map
            mapped = JOINT_NAME_MAP.get(npz_name, npz_name)
            if mapped == mj_name or npz_name == mj_name:
                mapping.append(npz_idx)
                found = True
                break
        if not found:
            print(f"  WARNING: MuJoCo joint '{mj_name}' not found in npz → using positional")
            return None

    print(f"  Joint mapping built successfully")
    return np.array(mapping)


def extract_trajectories(data):
    """Extract trajectory segments using split_points if available."""
    qpos = data["qpos"]
    qvel = data["qvel"]
    
    if "split_points" in data:
        sp = data["split_points"]
        if sp.ndim > 0 and len(sp) >= 2:
            trajs = []
            for i in range(0, len(sp) - 1, 2):
                start, end = int(sp[i]), int(sp[i+1])
                trajs.append({
                    "qpos": qpos[start:end],
                    "qvel": qvel[start:end],
                    "start": start,
                    "end": end,
                })
            if len(trajs) > 0:
                return trajs
    
    # Single trajectory = entire file
    return [{"qpos": qpos, "qvel": qvel, "start": 0, "end": len(qpos)}]


def replay_trajectory(model, mj_data, traj_qpos, traj_qvel, joint_mapping, 
                       freq, viewer=None, speed=1.0, collect=False):
    """
    Replay a single trajectory by setting qpos/qvel each frame.
    
    Returns collected data dict if collect=True.
    """
    n_steps = len(traj_qpos)
    dt = 1.0 / freq if freq > 0 else 0.02  # Default 50Hz
    
    n_mj_qpos = model.nq  # Total qpos dim in MuJoCo model
    n_mj_qvel = model.nv  # Total qvel dim in MuJoCo model
    n_npz_qpos = traj_qpos.shape[1] if traj_qpos.ndim > 1 else 0
    n_npz_qvel = traj_qvel.shape[1] if traj_qvel.ndim > 1 else 0
    
    print(f"  MuJoCo model: nq={n_mj_qpos}, nv={n_mj_qvel}")
    print(f"  NPZ data:     qpos_dim={n_npz_qpos}, qvel_dim={n_npz_qvel}")
    
    # Collected data storage
    collected = {
        "time": [],
        "qpos": [],
        "qvel": [],
        "base_pos": [],
        "base_quat": [],
        "base_lin_vel": [],
        "base_ang_vel": [],
        "joint_pos": [],
        "joint_vel": [],
        "com": [],
    } if collect else None
    
    # Number of joint DOFs in the npz (qpos has 7 root + N joints)
    n_npz_joints = n_npz_qpos - 7 if n_npz_qpos > 7 else 0
    n_mj_joints = n_mj_qpos - 7  # MuJoCo: free joint = 7 qpos
    
    for step in range(n_steps):
        qp = traj_qpos[step]
        qv = traj_qvel[step] if step < len(traj_qvel) else np.zeros(n_mj_qvel)
        
        # --- Set root (floating base) ---
        # qpos: [x, y, z, qw, qx, qy, qz, ...]
        # Some datasets might use [x,y,z, qx,qy,qz,qw] — check quaternion norm
        mj_data.qpos[:3] = qp[:3]  # Root position
        mj_data.qpos[3:7] = qp[3:7]  # Root quaternion (assumed wxyz MuJoCo convention)
        
        # --- Set joint positions ---
        if n_npz_joints > 0:
            npz_joint_pos = qp[7:7 + n_npz_joints]
            if joint_mapping is not None and len(joint_mapping) <= n_npz_joints:
                # Reorder from npz order to MuJoCo order
                for mj_idx, npz_idx in enumerate(joint_mapping):
                    if 7 + mj_idx < n_mj_qpos and npz_idx < n_npz_joints:
                        mj_data.qpos[7 + mj_idx] = npz_joint_pos[npz_idx]
            else:
                # Positional: copy min(npz_joints, mj_joints) 
                n_copy = min(n_npz_joints, n_mj_joints)
                mj_data.qpos[7:7 + n_copy] = npz_joint_pos[:n_copy]
        
        # --- Set velocities ---
        if n_npz_qvel > 0:
            mj_data.qvel[:3] = qv[:3]  # Root linear velocity
            mj_data.qvel[3:6] = qv[3:6]  # Root angular velocity
            
            n_npz_jvel = n_npz_qvel - 6
            n_mj_jvel = n_mj_qvel - 6
            if n_npz_jvel > 0:
                npz_joint_vel = qv[6:6 + n_npz_jvel]
                if joint_mapping is not None and len(joint_mapping) <= n_npz_jvel:
                    for mj_idx, npz_idx in enumerate(joint_mapping):
                        if 6 + mj_idx < n_mj_qvel and npz_idx < n_npz_jvel:
                            mj_data.qvel[6 + mj_idx] = npz_joint_vel[npz_idx]
                else:
                    n_copy = min(n_npz_jvel, n_mj_jvel)
                    mj_data.qvel[6:6 + n_copy] = npz_joint_vel[:n_copy]
        
        # Forward kinematics (compute positions, COMs etc. without stepping physics)
        mujoco.mj_forward(model, mj_data)
        
        # --- Collect data ---
        if collected is not None:
            t = step * dt
            collected["time"].append(t)
            collected["qpos"].append(mj_data.qpos.copy())
            collected["qvel"].append(mj_data.qvel.copy())
            collected["base_pos"].append(mj_data.qpos[:3].copy())
            collected["base_quat"].append(mj_data.qpos[3:7].copy())
            collected["base_lin_vel"].append(mj_data.qvel[:3].copy())
            collected["base_ang_vel"].append(mj_data.qvel[3:6].copy())
            collected["joint_pos"].append(mj_data.qpos[7:].copy())
            collected["joint_vel"].append(mj_data.qvel[6:].copy())
            collected["com"].append(mj_data.subtree_com[0].copy())  # Root body COM
        
        # --- Viewer sync ---
        if viewer is not None:
            _update_camera(viewer, mj_data.qpos[:3])
            viewer.sync()
            if speed > 0:
                time.sleep(dt / speed)
            while _viewer_state["paused"] and viewer.is_running() and not _viewer_state["restart"]:
                viewer.sync()
                time.sleep(0.05)
            if _viewer_state["restart"] or not viewer.is_running():
                break
        
        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"  Step {step}/{n_steps}  t={step*dt:.2f}s  "
                  f"pos=[{mj_data.qpos[0]:.3f}, {mj_data.qpos[1]:.3f}, {mj_data.qpos[2]:.3f}]",
                  end="\r")
    
    print()  # Newline after progress
    
    # Convert lists to arrays
    if collected is not None:
        for key in collected:
            collected[key] = np.array(collected[key])
    
    return collected


def save_collected_data(collected, output_path, mj_joint_names):
    """Save collected data to CSV and/or npz."""
    if output_path.endswith(".csv"):
        import csv
        
        n_joints = collected["joint_pos"].shape[1] if collected["joint_pos"].ndim > 1 else 0
        
        headers = ["time", "x", "y", "z", "qw", "qx", "qy", "qz",
                    "vx", "vy", "vz", "wx", "wy", "wz",
                    "com_x", "com_y", "com_z"]
        
        for i, name in enumerate(mj_joint_names[:n_joints]):
            headers.append(f"pos_{name}")
        for i, name in enumerate(mj_joint_names[:n_joints]):
            headers.append(f"vel_{name}")
        
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for i in range(len(collected["time"])):
                row = [collected["time"][i]]
                row.extend(collected["base_pos"][i])
                row.extend(collected["base_quat"][i])
                row.extend(collected["base_lin_vel"][i])
                row.extend(collected["base_ang_vel"][i])
                row.extend(collected["com"][i])
                row.extend(collected["joint_pos"][i][:n_joints])
                row.extend(collected["joint_vel"][i][:n_joints])
                writer.writerow(row)
        
        print(f"Saved CSV: {output_path} ({len(collected['time'])} rows)")
    
    else:
        # Save as npz
        np.savez_compressed(output_path, **collected)
        print(f"Saved NPZ: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Replay Booster T1 dataset in MuJoCo")
    parser.add_argument("--file", required=True, help="Motion filename (e.g. walking1.npz) or full path")
    parser.add_argument("--xml", default=_DEFAULT_XML, help="Path to MuJoCo XML model (default: resources/T1/T1_locomotion.xml)")
    parser.add_argument("--traj", type=int, default=None, help="Trajectory index to replay (default: all)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--realtime", action="store_true", help="Force real-time playback")
    parser.add_argument("--collect", action="store_true", help="Collect physics data during replay")
    parser.add_argument("--output", type=str, default=None, help="Output path for collected data (.csv or .npz)")
    parser.add_argument("--headless", action="store_true", help="No viewer (for headless data collection)")
    parser.add_argument("--loop", action="store_true", help="Loop the replay continuously")
    args = parser.parse_args()

    # Resolve file path: if just a filename, look in data/raw/
    filepath = args.file
    if not os.path.isabs(filepath) and not os.path.exists(filepath):
        filepath = os.path.join(_DEFAULT_DATA_DIR, filepath)
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        print(f"  Looked in: {_DEFAULT_DATA_DIR}")
        available = [f for f in os.listdir(_DEFAULT_DATA_DIR) if f.endswith(".npz")] if os.path.isdir(_DEFAULT_DATA_DIR) else []
        if available:
            print(f"  Available: {', '.join(sorted(available))}")
        sys.exit(1)

    # Load data
    print(f"Loading: {filepath}")
    data = np.load(filepath, allow_pickle=True)
    
    # Get frequency
    freq = 50.0  # Default
    if "frequency" in data:
        freq = float(data["frequency"].item() if data["frequency"].ndim == 0 else data["frequency"])
    print(f"Control frequency: {freq} Hz")
    
    # Extract trajectories
    trajs = extract_trajectories(data)
    print(f"Found {len(trajs)} trajectory segment(s)")
    for i, t in enumerate(trajs):
        dur = len(t["qpos"]) / freq if freq > 0 else 0
        print(f"  Traj {i}: frames {t['start']}–{t['end']} ({len(t['qpos'])} steps, {dur:.2f}s)")
    
    # Select trajectory
    if args.traj is not None:
        if args.traj >= len(trajs):
            print(f"ERROR: traj {args.traj} out of range (0–{len(trajs)-1})")
            sys.exit(1)
        trajs = [trajs[args.traj]]
        print(f"Selected trajectory {args.traj}")
    
    # Load MuJoCo model
    print(f"Loading MuJoCo model: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    mj_data = mujoco.MjData(model)
    
    # Get joint info
    mj_joint_names = get_mujoco_joint_names(model)
    print(f"MuJoCo model: nq={model.nq}, nv={model.nv}, joints={len(mj_joint_names)}")
    
    # Build joint mapping
    npz_joint_names = data["joint_names"] if "joint_names" in data else None
    joint_mapping = build_joint_mapping(npz_joint_names, mj_joint_names)
    
    # Playback speed
    speed = args.speed
    if args.realtime:
        speed = 1.0
    if args.headless:
        speed = 0  # No delays
    
    # Run replay
    all_collected = []
    
    if args.headless:
        # Headless mode
        print("\n--- Headless replay (no viewer) ---")
        for i, traj in enumerate(trajs):
            print(f"\nTrajectory {i}:")
            collected = replay_trajectory(
                model, mj_data, traj["qpos"], traj["qvel"],
                joint_mapping, freq, viewer=None, speed=0, collect=args.collect
            )
            if collected:
                all_collected.append(collected)
    else:
        # Viewer mode
        print(f"\n--- Launching MuJoCo viewer (speed={speed}x) ---")
        print("Controls: [Space] pause/resume, [R] restart, [O] toggle camera, [Esc] quit")
        print("Contact forces: [F] or Right-click → Rendering → 'Contact Force' / 'Contact Point'")

        with mujoco.viewer.launch_passive(model, mj_data, key_callback=_key_callback) as viewer:
            # Set initial isometric camera
            _update_camera(viewer, mj_data.qpos[:3])
            running = True
            while running:
                _viewer_state["restart"] = False
                for i, traj in enumerate(trajs):
                    print(f"\nTrajectory {i}:")
                    collected = replay_trajectory(
                        model, mj_data, traj["qpos"], traj["qvel"],
                        joint_mapping, freq, viewer=viewer, speed=speed,
                        collect=args.collect
                    )
                    if collected:
                        all_collected.append(collected)

                    if not viewer.is_running():
                        running = False
                        break
                    if _viewer_state["restart"]:
                        break

                if _viewer_state["restart"]:
                    continue

                if not args.loop:
                    # Keep viewer open after replay — R restarts, Esc quits
                    print("\nReplay complete. [R] replay, [Esc] quit.")
                    while viewer.is_running() and not _viewer_state["restart"]:
                        _update_camera(viewer, mj_data.qpos[:3])
                        viewer.sync()
                        time.sleep(0.05)
                    if _viewer_state["restart"]:
                        continue
                    break
    
    # Save collected data
    if args.collect and all_collected:
        output = args.output or filepath.replace(".npz", "_collected.csv")
        
        if len(all_collected) == 1:
            save_collected_data(all_collected[0], output, mj_joint_names)
        else:
            # Save each trajectory separately
            base, ext = os.path.splitext(output)
            for i, c in enumerate(all_collected):
                path = f"{base}_traj{i}{ext}"
                save_collected_data(c, path, mj_joint_names)
    
    data.close()
    print("Done.")


if __name__ == "__main__":
    main()
