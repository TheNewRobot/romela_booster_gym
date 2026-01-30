import isaacgym
import os
import yaml
import argparse
import time
import signal
import imageio
import torch
from envs import *
from policies.actor_critic import ActorCritic
from utils.isaac_gym_utils import get_friction, set_friction, print_friction, print_play_status
from utils.config_loader import load_config

# === CONFIGURABLE PARAMETERS ===
FRICTION_OVERRIDE = None  # None=use config, float=override foot friction

# Leg joint indices (T1 robot: joints 11-22 are leg joints)
LEG_JOINT_START = 11
LEG_JOINT_END = 23
NUM_LEG_JOINTS = 12


def load_policy(policy_path, cfg, device):
    """Load policy - supports both .pth (ActorCritic) and .pt (JIT)."""
    is_jit = policy_path.endswith(".pt")
    
    if is_jit:
        policy = torch.jit.load(policy_path, map_location=device)
        policy.eval()
        return policy, True
    else:
        model = ActorCritic(cfg["env"]["num_actions"], cfg["env"]["num_observations"], cfg["env"]["num_privileged_obs"])
        model_dict = torch.load(policy_path, map_location="cpu", weights_only=True)
        model.load_state_dict(model_dict["model"])
        model = model.to(device)
        model.eval()
        return model, False


def build_obs_47(env, actions, cfg):
    """Build 47-dim observation for leg-only policies."""
    obs = torch.zeros(env.num_envs, 47, dtype=torch.float32, device=env.device)
    obs[:, 0:3] = env.projected_gravity * cfg["normalization"]["gravity"]
    obs[:, 3:6] = env.base_ang_vel * cfg["normalization"]["ang_vel"]
    gait_active = (env.gait_frequency > 1.0e-8).float().unsqueeze(-1)
    obs[:, 6] = env.commands[:, 0] * cfg["normalization"]["lin_vel"] * gait_active.squeeze()
    obs[:, 7] = env.commands[:, 1] * cfg["normalization"]["lin_vel"] * gait_active.squeeze()
    obs[:, 8] = env.commands[:, 2] * cfg["normalization"]["ang_vel"] * gait_active.squeeze()
    obs[:, 9] = torch.cos(2 * torch.pi * env.gait_process) * gait_active.squeeze()
    obs[:, 10] = torch.sin(2 * torch.pi * env.gait_process) * gait_active.squeeze()
    obs[:, 11:23] = (env.dof_pos[:, LEG_JOINT_START:LEG_JOINT_END] - env.default_dof_pos[:, LEG_JOINT_START:LEG_JOINT_END]) * cfg["normalization"]["dof_pos"]
    obs[:, 23:35] = env.dof_vel[:, LEG_JOINT_START:LEG_JOINT_END] * cfg["normalization"]["dof_vel"]
    obs[:, 35:47] = actions
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Task config name")
    parser.add_argument("--policy", type=str, default=None, help="Path to policy (.pth or .pt). If not provided, uses checkpoint from config.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    args = parser.parse_args()

    # Load config
    cfg_file = None
    for root, dirs, files in os.walk("envs"):
        for file in files:
            if file == f"{args.task}.yaml":
                cfg_file = os.path.join(root, file)
                break
    if cfg_file is None:
        raise FileNotFoundError(f"Config file '{args.task}.yaml' not found in envs/")
    
    print(f"Loading config from: {cfg_file}")
    cfg = load_config(cfg_file)
    
    # Override for play mode
    cfg["env"]["num_envs"] = args.num_envs
    cfg["env"]["play"] = True
    cfg["terrain"]["type"] = "plane"
    cfg["basic"]["headless"] = False

    device = cfg.get("basic", {}).get("rl_device", "cuda:0")

    # Create environment
    task_name = cfg.get("basic", {}).get("task", args.task)
    print(f"Creating environment: {task_name}")
    task_class = eval(task_name)
    env = task_class(cfg)
    device = env.device

    # Determine policy path
    policy_path = args.policy
    if policy_path is None:
        policy_path = cfg["basic"].get("checkpoint")
        if policy_path is None:
            raise ValueError("No policy provided. Use --policy or set checkpoint in config.")
    
    print(f"Loading policy from: {policy_path}")
    policy, is_jit = load_policy(policy_path, cfg, device)
    print(f"Policy type: {'JIT' if is_jit else 'ActorCritic'}")

    # Check policy dimensions
    num_policy_actions = cfg["env"]["num_actions"]
    legs_only = (num_policy_actions == NUM_LEG_JOINTS)
    env_num_actions = env.num_actions
    print(f"Policy outputs {num_policy_actions} actions ({'leg joints only' if legs_only else 'all joints'})")
    print(f"Environment expects {env_num_actions} actions")

    # Friction handling
    if FRICTION_OVERRIDE is not None:
        set_friction(env, FRICTION_OVERRIDE)
        print_friction(env, overridden=True)
    else:
        print_friction(env, overridden=False)

    print("="*50)
    print("Interactive Play Mode")
    print("="*50)
    print("Controls:")
    print("  [w/s]   : forward/backward (vx)")
    print("  [a/d]   : strafe left/right (vy)")
    print("  [q/e]   : turn left/right (vyaw)")
    print("  [r]     : reset episode")
    print("  [space] : play/pause")
    print("="*50)
    print("Press [space] to start")

    # Initialize
    obs, infos = env.reset()
    obs = obs.to(device)
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.render()
    
    actions = torch.zeros(env.num_envs, num_policy_actions, dtype=torch.float32, device=device)
    cmd = env.commands[0, :3].cpu().numpy()
    print_play_status({"is_playing": env.is_playing}, cmd)
    
    step_count = 0
    while True:
        env.render()
        
        if env.reset_triggered:
            obs, infos = env.reset()
            obs = obs.to(device)
            actions.zero_()
            env.reset_triggered = False
            step_count = 0
            print("\rEpisode reset                                        ")
            cmd = env.commands[0, :3].cpu().numpy()
            print_play_status({"is_playing": env.is_playing}, cmd)
        
        if not env.is_playing:
            continue
        
        with torch.no_grad():
            # Get observation
            if legs_only:
                obs_input = obs if obs.shape[-1] == 47 else build_obs_47(env, actions, cfg)
            else:
                obs_input = obs
            
            # Get actions from policy
            if is_jit:
                actions[:] = policy(obs_input)
            else:
                dist = policy.act(obs_input)
                actions[:] = dist.loc
            
            actions[:] = torch.clip(actions, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
            
            # Handle action dimension mismatch
            if legs_only and env_num_actions != NUM_LEG_JOINTS:
                env_actions = torch.zeros(env.num_envs, env_num_actions, dtype=torch.float32, device=device)
                env_actions[:, LEG_JOINT_START:LEG_JOINT_END] = actions
            else:
                env_actions = actions
            
            # Step environment
            obs, rew, done, infos = env.step(env_actions)
            obs = obs.to(device)
        
        step_count += 1
        if step_count % 10 == 0:
            cmd = env.commands[0, :3].cpu().numpy()
            print_play_status({"is_playing": env.is_playing}, cmd)


if __name__ == "__main__":
    main()