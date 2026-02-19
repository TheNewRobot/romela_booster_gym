"""Inspect .npz motion data files (shapes, joint names, metadata)."""
import os
import sys
import numpy as np
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_T1_DIR = os.path.dirname(_SCRIPT_DIR)
_DEFAULT_DATA_DIR = os.path.join(_T1_DIR, "data", "raw")

def inspect_npz(filepath):
    data = np.load(filepath, allow_pickle=True)
    
    print(f"=== {filepath} ===")
    print(f"Keys: {list(data.keys())}\n")
    
    for key in sorted(data.keys()):
        arr = data[key]
        if arr.ndim == 0:
            # Scalar or object array
            val = arr.item()
            print(f"  {key:20s}  scalar = {val}")
        else:
            print(f"  {key:20s}  shape={str(arr.shape):20s}  dtype={arr.dtype}")
    
    # Key details
    print("\n--- Key Details ---")
    
    if "qpos" in data:
        qpos = data["qpos"]
        print(f"\nqpos: {qpos.shape}")
        print(f"  Timesteps: {qpos.shape[0]}")
        print(f"  Dim per step: {qpos.shape[1] if qpos.ndim > 1 else 'N/A'}")
        print(f"  Layout: [x,y,z(3) + quat(4) + joint_pos(N)]")
        if qpos.ndim > 1:
            print(f"  → Num joints (qpos): {qpos.shape[1] - 7}")
            print(f"  Root pos  range: {qpos[:, :3].min(axis=0)} to {qpos[:, :3].max(axis=0)}")
            print(f"  Root quat[0]:    {qpos[0, 3:7]}")
    
    if "qvel" in data:
        qvel = data["qvel"]
        print(f"\nqvel: {qvel.shape}")
        print(f"  Layout: [lin_vel(3) + ang_vel(3) + joint_vel(N)]")
        if qvel.ndim > 1:
            print(f"  → Num joints (qvel): {qvel.shape[1] - 6}")
    
    if "joint_names" in data:
        names = data["joint_names"]
        if names.ndim == 0:
            names = names.item()
        print(f"\nJoint names: {names}")
    
    if "frequency" in data:
        freq = data["frequency"].item() if data["frequency"].ndim == 0 else data["frequency"]
        print(f"\nControl frequency: {freq} Hz")
    
    if "split_points" in data:
        sp = data["split_points"]
        print(f"\nSplit points: {sp}")
        if sp.ndim > 0 and len(sp) >= 2:
            for i in range(0, len(sp) - 1, 2):
                start, end = int(sp[i]), int(sp[i+1])
                print(f"  Trajectory {i//2}: frames {start}–{end} ({end - start} steps)")
    
    if "njnt" in data:
        njnt = data["njnt"].item() if data["njnt"].ndim == 0 else data["njnt"]
        print(f"\nNum joints (njnt): {njnt}")
    
    if "jnt_type" in data:
        jt = data["jnt_type"]
        print(f"Joint types: {jt}  (0=free/root, 3=hinge)")
    
    # Duration estimate
    if "qpos" in data and "frequency" in data:
        freq = data["frequency"].item() if data["frequency"].ndim == 0 else float(data["frequency"])
        if freq > 0:
            duration = qpos.shape[0] / freq
            print(f"\nEstimated duration: {duration:.2f}s at {freq}Hz")

    data.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Filename (e.g. walking1.npz) or full path")
    args = parser.parse_args()

    filepath = args.file
    if not os.path.isabs(filepath) and not os.path.exists(filepath):
        filepath = os.path.join(_DEFAULT_DATA_DIR, filepath)
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        available = [f for f in os.listdir(_DEFAULT_DATA_DIR) if f.endswith(".npz")] if os.path.isdir(_DEFAULT_DATA_DIR) else []
        if available:
            print(f"  Available: {', '.join(sorted(available))}")
        sys.exit(1)

    inspect_npz(filepath)
