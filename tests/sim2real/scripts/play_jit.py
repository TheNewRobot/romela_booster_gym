import isaacgym
import os
import yaml
import argparse
import numpy as np
import torch
from envs import *

# Leg joint indices (T1 robot: joints 11-22 are leg joints)
LEG_JOINT_START = 11
LEG_JOINT_END = 23
NUM_LEG_JOINTS = 12

def build_obs_47(env, actions, cfg):
    """Build 47-dim observation matching deploy/policy.py format.
    
    Layout:
    [0:3]   projected_gravity (3)
    [3:6]   base_ang_vel (3)
    [6:9]   commands vx,vy,vyaw (3)
    [9:11]  gait_phase cos,sin (2)
    [11:23] dof_pos_delta for legs (12)
    [23:35] dof_vel for legs (12)
    [35:47] previous actions (12)
    """
    obs = torch.zeros(env.num_envs, 47, dtype=torch.float32, device=env.device)
    
    # [0:3] projected gravity
    obs[:, 0:3] = env.projected_gravity * cfg["normalization"]["gravity"]
    
    # [3:6] base angular velocity  
    obs[:, 3:6] = env.base_ang_vel * cfg["normalization"]["ang_vel"]
    
    # [6:9] commands (vx, vy, vyaw) - only active if gait_frequency > 0
    gait_active = (env.gait_frequency > 1.0e-8).float().unsqueeze(-1)
    obs[:, 6] = env.commands[:, 0] * cfg["normalization"]["lin_vel"] * gait_active.squeeze()
    obs[:, 7] = env.commands[:, 1] * cfg["normalization"]["lin_vel"] * gait_active.squeeze()
    obs[:, 8] = env.commands[:, 2] * cfg["normalization"]["ang_vel"] * gait_active.squeeze()
    
    # [9:11] gait phase (cos, sin)
    obs[:, 9] = torch.cos(2 * np.pi * env.gait_process) * gait_active.squeeze()
    obs[:, 10] = torch.sin(2 * np.pi * env.gait_process) * gait_active.squeeze()
    
    # [11:23] dof_pos delta for LEG joints only (indices 11:23)
    obs[:, 11:23] = (env.dof_pos[:, LEG_JOINT_START:LEG_JOINT_END] - env.default_dof_pos[:, LEG_JOINT_START:LEG_JOINT_END]) * cfg["normalization"]["dof_pos"]
    
    # [23:35] dof_vel for LEG joints only (indices 11:23)  
    obs[:, 23:35] = env.dof_vel[:, LEG_JOINT_START:LEG_JOINT_END] * cfg["normalization"]["dof_vel"]
    
    # [35:47] previous actions
    obs[:, 35:47] = actions
    
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Task config name (e.g., t1_frd)")
    parser.add_argument("--policy", required=True, type=str, help="Path to JIT policy (.pt)")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda:0 or cpu)")
    args = parser.parse_args()

    # Load config - search in envs/ directory
    cfg_file = None
    for root, dirs, files in os.walk("envs"):
        for file in files:
            if file == f"{args.task}.yaml":
                cfg_file = os.path.join(root, file)
                break
    if cfg_file is None:
        raise FileNotFoundError(f"Config file '{args.task}.yaml' not found in envs/")
    
    print(f"Loading config from: {cfg_file}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # Override for play mode
    if "env" not in cfg:
        cfg["env"] = {}
    cfg["env"]["num_envs"] = args.num_envs
    cfg["env"]["play"] = True
    
    if "terrain" not in cfg:
        cfg["terrain"] = {}
    cfg["terrain"]["type"] = "plane"
    
    if "basic" not in cfg:
        cfg["basic"] = {}
    cfg["basic"]["headless"] = False
    
    # Get device from config or use default
    device = cfg.get("basic", {}).get("rl_device", "cuda:0")
    if args.device:
        device = args.device
        cfg["basic"]["sim_device"] = args.device
        cfg["basic"]["rl_device"] = args.device

    # Load JIT policy
    print(f"Loading policy from: {args.policy}")
    policy = torch.jit.load(args.policy, map_location=device)
    policy.eval()
    
    # Verify policy dimensions
    test_obs = torch.zeros(1, 47, device=device)
    test_out = policy(test_obs)
    print(f"Policy: 47 obs -> {test_out.shape[1]} actions")
    assert test_out.shape[1] == NUM_LEG_JOINTS, f"Expected {NUM_LEG_JOINTS} actions, got {test_out.shape[1]}"
    
    # Create environment - task class name from config or from args
    task_name = cfg.get("basic", {}).get("task", args.task)
    print(f"Creating environment: {task_name}")
    task_class = eval(task_name)
    env = task_class(cfg)
    device = env.device
    
    # Check if env expects 12 or 23 actions
    env_num_actions = env.num_actions
    print(f"Environment expects {env_num_actions} actions, policy outputs {NUM_LEG_JOINTS}")
    
    # Initialize actions buffer (12 leg joint actions)
    actions = torch.zeros(env.num_envs, NUM_LEG_JOINTS, dtype=torch.float32, device=device)
    
    print("\n" + "="*50)
    print("Play JIT Policy - Interactive Mode")
    print("="*50)
    print("Controls:")
    print("  [w/s]   : forward/backward (vx)")
    print("  [a/d]   : strafe left/right (vy)")
    print("  [q/e]   : turn left/right (vyaw)")
    print("  [r]     : reset episode")
    print("  [space] : play/pause")
    print("="*50)
    print("Press [space] to start\n")
    
    obs, infos = env.reset()
    print(f"Observation shape from env: {obs.shape}")
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.render()
    
    step_count = 0
    while True:
        env.render()
        
        if env.reset_triggered:
            obs, infos = env.reset()
            actions.zero_()
            env.reset_triggered = False
            step_count = 0
            print("Episode reset")
        
        if not env.is_playing:
            continue
        
        with torch.no_grad():
            # Use env's observation directly (computed by _compute_observations)
            # The env already builds the correct 47-dim observation
            obs_input = obs if obs.shape[-1] == 47 else build_obs_47(env, actions, cfg)
            
            # Run policy to get 12 leg joint actions
            actions[:] = policy(obs_input)
            actions[:] = torch.clip(actions, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
            
            # Handle action dimension mismatch if needed
            if env_num_actions == NUM_LEG_JOINTS:
                # Env expects 12 actions (leg joints only)
                env_actions = actions
            else:
                # Env expects full 23 actions - pad with zeros
                env_actions = torch.zeros(env.num_envs, env_num_actions, dtype=torch.float32, device=device)
                env_actions[:, LEG_JOINT_START:LEG_JOINT_END] = actions
            
            # Step environment
            obs, rew, done, infos = env.step(env_actions)
            step_count += 1
            
            # Print status periodically
            if step_count % 100 == 0:
                cmd = env.commands[0, :3].cpu().numpy()
                vel = env.base_lin_vel[0].cpu().numpy()
                act_range = (actions[0].min().item(), actions[0].max().item())
                print(f"Step {step_count} | Cmd: [{cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}] | Vel: [{vel[0]:.2f}, {vel[1]:.2f}] | Act: [{act_range[0]:.2f}, {act_range[1]:.2f}]")


if __name__ == "__main__":
    main()