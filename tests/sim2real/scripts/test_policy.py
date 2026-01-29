import argparse
import torch
import numpy as np

NUM_OBS = 47
NUM_ACTIONS = 12

def main():
    parser = argparse.ArgumentParser(description="Test policy loading and inference")
    parser.add_argument("--policy", type=str, default="models/t1_frd.pt", help="Path to JIT policy")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Phase 1: Policy Loading Test")
    print(f"{'='*60}")

    # Load policy
    print(f"\n[1] Loading policy from: {args.policy}")
    try:
        policy = torch.jit.load(args.policy)
        policy.eval()
        print("    ✓ Policy loaded successfully")
    except Exception as e:
        print(f"    ✗ Failed to load policy: {e}")
        return

    # Print model structure
    print(f"\n[2] Model structure:")
    print(f"    {policy}")

    # Test single inference
    print(f"\n[3] Single inference test:")
    dummy_obs = torch.zeros(1, NUM_OBS, dtype=torch.float32)
    print(f"    Input shape:  {dummy_obs.shape} (expected: [1, {NUM_OBS}])")
    try:
        with torch.no_grad():
            actions = policy(dummy_obs)
        print(f"    Output shape: {actions.shape} (expected: [1, {NUM_ACTIONS}])")
        if actions.shape == (1, NUM_ACTIONS):
            print("    ✓ Dimensions match")
        else:
            print(f"    ✗ Dimension mismatch! Expected [1, {NUM_ACTIONS}]")
            return
    except Exception as e:
        print(f"    ✗ Inference failed: {e}")
        return

    # Test batch inference
    print(f"\n[4] Batch inference test:")
    batch_size = 16
    batch_obs = torch.randn(batch_size, NUM_OBS, dtype=torch.float32)
    print(f"    Input shape:  {batch_obs.shape}")
    try:
        with torch.no_grad():
            batch_actions = policy(batch_obs)
        print(f"    Output shape: {batch_actions.shape} (expected: [{batch_size}, {NUM_ACTIONS}])")
        if batch_actions.shape == (batch_size, NUM_ACTIONS):
            print("    ✓ Batch inference works")
        else:
            print(f"    ✗ Dimension mismatch!")
            return
    except Exception as e:
        print(f"    ✗ Batch inference failed: {e}")
        return

    # Test with realistic observation values
    print(f"\n[5] Realistic observation test:")
    obs = torch.zeros(1, NUM_OBS, dtype=torch.float32)
    obs[0, 0:3] = torch.tensor([0.0, 0.0, -1.0])  # projected gravity (standing)
    obs[0, 3:6] = torch.tensor([0.0, 0.0, 0.0])   # base angular velocity
    obs[0, 6:9] = torch.tensor([0.3, 0.0, 0.0])   # commands (vx=0.3)
    obs[0, 9:11] = torch.tensor([1.0, 0.0])       # gait phase (cos, sin)
    # obs[11:23] = dof_pos delta (zeros = at default)
    # obs[23:35] = dof_vel (zeros = stationary)
    # obs[35:47] = previous actions (zeros)
    print(f"    Observation breakdown:")
    print(f"      [0:3]   projected_gravity: {obs[0, 0:3].tolist()}")
    print(f"      [3:6]   base_ang_vel:      {obs[0, 3:6].tolist()}")
    print(f"      [6:9]   commands (vx,vy,vyaw): {obs[0, 6:9].tolist()}")
    print(f"      [9:11]  gait_phase (cos,sin): {obs[0, 9:11].tolist()}")
    print(f"      [11:23] dof_pos_delta:     zeros")
    print(f"      [23:35] dof_vel:           zeros")
    print(f"      [35:47] prev_actions:      zeros")
    
    with torch.no_grad():
        actions = policy(obs)
    print(f"    Output actions: {actions[0].tolist()}")
    print(f"    Action range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
    print("    ✓ Realistic inference complete")

    print(f"\n{'='*60}")
    print("Phase 1 Complete: Policy is ready for replay")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()