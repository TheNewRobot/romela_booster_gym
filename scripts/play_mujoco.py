import os
import sys
import glob
import yaml
import argparse
import numpy as np
import torch
import mujoco, mujoco.viewer
from policies.actor_critic import *

LEG_JOINT_START = 11
LEG_JOINT_END = 23
NUM_LEG_JOINTS = 12


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


def offset_to_spherical(offset):
    """Convert [x, y, z] offset to MuJoCo camera spherical coords."""
    x, y, z = offset
    distance = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arcsin(z / distance)) if distance > 0 else 0
    return distance, azimuth, elevation


def find_experiment_config(policy_path):
    """Auto-detect config.yaml from policy's experiment folder."""
    policy_dir = os.path.dirname(os.path.abspath(policy_path))
    
    # If policy is in nn/ subfolder, go up one level
    if os.path.basename(policy_dir) == "nn":
        experiment_dir = os.path.dirname(policy_dir)
        config_path = os.path.join(experiment_dir, "config.yaml")
        if os.path.exists(config_path):
            return config_path
    return None


def load_policy(policy_path, cfg):
    """Load policy - supports both .pth (ActorCritic) and .pt (JIT)."""
    is_jit = policy_path.endswith(".pt")
    
    if is_jit:
        policy = torch.jit.load(policy_path, map_location="cpu")
        policy.eval()
        return policy, True
    else:
        model = ActorCritic(cfg["env"]["num_actions"], cfg["env"]["num_observations"], cfg["env"]["num_privileged_obs"])
        model_dict = torch.load(policy_path, map_location="cpu", weights_only=True)
        model.load_state_dict(model_dict["model"])
        model.eval()
        return model, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, help="Task config name (fallback if auto-detect fails)")
    parser.add_argument("--policy", required=True, type=str, help="Path to policy (.pth or .pt)")
    parser.add_argument("--config", type=str, default=None, help="Explicit config path (auto-detected if not provided)")
    args = parser.parse_args()

    # Config resolution: explicit --config > auto-detect from policy > fallback to --task
    cfg_file = None
    if args.config:
        cfg_file = args.config
        print(f"Using explicit config: {cfg_file}")
    else:
        cfg_file = find_experiment_config(args.policy)
        if cfg_file:
            print(f"Auto-detected config: {cfg_file}")
        elif args.task:
            cfg_file = os.path.join("envs", "locomotion", f"{args.task}.yaml")
            print(f"Fallback to task config: {cfg_file}")
        else:
            raise ValueError("Could not find config. Provide --config or --task, or ensure config.yaml exists in experiment folder.")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    print(f"Loading policy from {args.policy}")
    policy, is_jit = load_policy(args.policy, cfg)
    print(f"Policy type: {'JIT' if is_jit else 'ActorCritic'}")

    mj_model = mujoco.MjModel.from_xml_path(cfg["asset"]["mujoco_file"])
    mj_model.opt.timestep = cfg["sim"]["dt"]
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    
    # Initialize default positions and PD gains for all actuators
    num_actuators = mj_model.nu
    default_dof_pos = np.zeros(num_actuators, dtype=np.float32)
    dof_stiffness = np.zeros(num_actuators, dtype=np.float32)
    dof_damping = np.zeros(num_actuators, dtype=np.float32)
    
    for i in range(num_actuators):
        actuator_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        
        # Default joint angles
        found = False
        for name in cfg["init_state"]["default_joint_angles"].keys():
            if name in actuator_name:
                default_dof_pos[i] = cfg["init_state"]["default_joint_angles"][name]
                found = True
                break
        if not found:
            default_dof_pos[i] = cfg["init_state"]["default_joint_angles"]["default"]

        # PD gains
        found = False
        for name in cfg["control"]["stiffness"].keys():
            if name in actuator_name:
                dof_stiffness[i] = cfg["control"]["stiffness"][name]
                dof_damping[i] = cfg["control"]["damping"][name]
                found = True
                break
        if not found:
            raise ValueError(f"PD gain of joint {actuator_name} were not defined")

    # Set initial pose
    mj_data.qpos = np.concatenate([
        np.array(cfg["init_state"]["pos"], dtype=np.float32),
        np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=np.float32),
        default_dof_pos,
    ])
    mujoco.mj_forward(mj_model, mj_data)

    # Determine if policy controls all joints or just legs
    num_policy_actions = cfg["env"]["num_actions"]
    legs_only = (num_policy_actions == NUM_LEG_JOINTS)
    print(f"Policy outputs {num_policy_actions} actions ({'leg joints only' if legs_only else 'all joints'})")
    print(f"MuJoCo model has {num_actuators} actuators")
    
    # Determine joint indexing based on MuJoCo model
    if num_actuators == NUM_LEG_JOINTS:
        # MuJoCo model only has leg actuators
        mj_leg_start, mj_leg_end = 0, NUM_LEG_JOINTS
    else:
        # MuJoCo model has all joints
        mj_leg_start, mj_leg_end = LEG_JOINT_START, LEG_JOINT_END

    actions = np.zeros(num_policy_actions, dtype=np.float32)
    dof_targets = np.copy(default_dof_pos)
    gait_frequency = gait_process = 0.0
    it = 0
    
    # Viewer state
    state = {
        "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
        "reset": False,
        "is_playing": False,
        "camera_follow": True,
    }
    cmd_increment = {"vx": 0.2, "vy": 0.2, "vyaw": 0.3}
    cmd_limits = {"vx": (-1.0, 1.0), "vy": (-1.0, 1.0), "vyaw": (-1.0, 1.0)}
    
    # Camera settings from config
    cam_offset = [-2.0, -2.0, -0.5]  # [x, y, z]: negative x = in front, z = above
    cam_distance, cam_azimuth, cam_elevation = offset_to_spherical(cam_offset)
    
    # Store initial pose for reset
    initial_qpos = mj_data.qpos.copy()
    initial_qvel = np.zeros_like(mj_data.qvel)

    def print_status():
        mode = "PLAY" if state["is_playing"] else "PAUSE"
        cam = "FOLLOW" if state["camera_follow"] else "FREE"
        print(f"\r[{mode}] [{cam}] Cmd: vx={state['vx']:+.2f}  vy={state['vy']:+.2f}  vyaw={state['vyaw']:+.2f}    ", end="", flush=True)

    def key_callback(keycode):
        # GLFW key codes: Arrow Up=265, Down=264, Left=263, Right=262, Q=81, E=69, R=82, Space=32, O=79
        if keycode == 265:  # Arrow Up - forward
            state["vx"] = min(state["vx"] + cmd_increment["vx"], cmd_limits["vx"][1])
        elif keycode == 264:  # Arrow Down - backward
            state["vx"] = max(state["vx"] - cmd_increment["vx"], cmd_limits["vx"][0])
        elif keycode == 263:  # Arrow Left - strafe left
            state["vy"] = min(state["vy"] + cmd_increment["vy"], cmd_limits["vy"][1])
        elif keycode == 262:  # Arrow Right - strafe right
            state["vy"] = max(state["vy"] - cmd_increment["vy"], cmd_limits["vy"][0])
        elif keycode == 81:  # Q - turn left
            state["vyaw"] = min(state["vyaw"] + cmd_increment["vyaw"], cmd_limits["vyaw"][1])
        elif keycode == 69:  # E - turn right
            state["vyaw"] = max(state["vyaw"] - cmd_increment["vyaw"], cmd_limits["vyaw"][0])
        elif keycode == 32:  # Space - pause/unpause
            state["is_playing"] = not state["is_playing"]
        elif keycode == 82:  # R - reset
            state["reset"] = True
            state["vx"] = 0.0
            state["vy"] = 0.0
            state["vyaw"] = 0.0
        elif keycode == 79:  # O - toggle camera follow
            state["camera_follow"] = not state["camera_follow"]
        
        print_status()

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        # Set initial camera position
        robot_pos = mj_data.qpos[:3]
        viewer.cam.lookat[:] = robot_pos
        viewer.cam.distance = cam_distance
        viewer.cam.azimuth = cam_azimuth
        viewer.cam.elevation = cam_elevation
        
        print("="*50)
        print("Play MuJoCo - Keyboard Controls")
        print("="*50)
        print("  [↑/↓]   : forward/backward")
        print("  [←/→]   : strafe left/right")
        print("  [Q/E]   : turn left/right")
        print("  [Space] : pause/unpause simulation")
        print("  [R]     : reset robot pose")
        print("  [O]     : toggle camera follow")
        print("="*50)
        print("Press [Space] to start")
        print_status()
        
        while viewer.is_running():
            # Handle reset
            if state["reset"]:
                mj_data.qpos[:] = initial_qpos
                mj_data.qvel[:] = initial_qvel
                mujoco.mj_forward(mj_model, mj_data)
                actions.fill(0)
                dof_targets[:] = default_dof_pos
                gait_process = 0.0
                state["reset"] = False
                print(f"\rRobot reset!")
                print_status()
            
            # Update camera if following
            if state["camera_follow"]:
                robot_pos = mj_data.qpos[:3]
                viewer.cam.lookat[:] = robot_pos
                viewer.cam.distance = cam_distance
                viewer.cam.azimuth = cam_azimuth
                viewer.cam.elevation = cam_elevation
            
            # Skip physics if paused
            if not state["is_playing"]:
                viewer.sync()
                continue

            # Update gait frequency based on commands
            lin_vel_x, lin_vel_y, ang_vel_yaw = state["vx"], state["vy"], state["vyaw"]
            if lin_vel_x == 0 and lin_vel_y == 0 and ang_vel_yaw == 0:
                gait_frequency = 0
            else:
                gait_frequency = np.average(cfg["commands"]["gait_frequency"])

            # Get robot state
            dof_pos = mj_data.qpos.astype(np.float32)[7:]
            dof_vel = mj_data.qvel.astype(np.float32)[6:]
            quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
            base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

            # Run policy at decimated rate
            if it % cfg["control"]["decimation"] == 0:
                obs = np.zeros(cfg["env"]["num_observations"], dtype=np.float32)
                obs[0:3] = projected_gravity * cfg["normalization"]["gravity"]
                obs[3:6] = base_ang_vel * cfg["normalization"]["ang_vel"]
                obs[6] = lin_vel_x * cfg["normalization"]["lin_vel"] * (gait_frequency > 1.0e-8)
                obs[7] = lin_vel_y * cfg["normalization"]["lin_vel"] * (gait_frequency > 1.0e-8)
                obs[8] = ang_vel_yaw * cfg["normalization"]["ang_vel"] * (gait_frequency > 1.0e-8)
                obs[9] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[10] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                
                if legs_only:
                    obs[11:23] = (dof_pos[mj_leg_start:mj_leg_end] - default_dof_pos[mj_leg_start:mj_leg_end]) * cfg["normalization"]["dof_pos"]
                    obs[23:35] = dof_vel[mj_leg_start:mj_leg_end] * cfg["normalization"]["dof_vel"]
                else:
                    obs[11:23] = (dof_pos - default_dof_pos) * cfg["normalization"]["dof_pos"]
                    obs[23:35] = dof_vel * cfg["normalization"]["dof_vel"]
                obs[35:47] = actions

                # Get actions from policy
                obs_tensor = torch.tensor(obs).unsqueeze(0)
                if is_jit:
                    actions[:] = policy(obs_tensor).detach().numpy().squeeze()
                else:
                    dist = policy.act(obs_tensor)
                    actions[:] = dist.loc.detach().numpy().squeeze()
                actions[:] = np.clip(actions, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])

                # Apply actions to targets
                dof_targets[:] = default_dof_pos
                if legs_only:
                    dof_targets[mj_leg_start:mj_leg_end] += cfg["control"]["action_scale"] * actions
                else:
                    dof_targets += cfg["control"]["action_scale"] * actions

            # PD control
            mj_data.ctrl = np.clip(
                dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
                mj_model.actuator_ctrlrange[:, 0],
                mj_model.actuator_ctrlrange[:, 1],
            )
            
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            it += 1
            gait_process = np.fmod(gait_process + cfg["sim"]["dt"] * gait_frequency, 1.0)