import os
import yaml
import argparse
import numpy as np
import torch
import mujoco, mujoco.viewer
from utils.config_loader import find_experiment_config, load_config
from utils.policy_loader import load_policy
# sim2real utilities for applying calibrated joint dynamics
from tests.sim2real.utils.mujoco_utils import get_joint_dof_indices, apply_sim_params

# === CONFIGURABLE PARAMETERS ===
CAM_OFFSET = [-2.0, -2.0, -0.5]  # [x, y, z] camera offset for chase mode
FRICTION_OVERRIDE = None  # None=default, float=sliding only, [a,b,c]=all three

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, help="Task config name (fallback if auto-detect fails)")
    parser.add_argument("--policy", required=True, type=str, help="Path to policy (.pth or .pt)")
    parser.add_argument("--config", type=str, default=None, help="Explicit config path (auto-detected if not provided)")
    parser.add_argument("--sim-params", type=str, nargs='?', const="tests/sim2real/config/sim_params.yaml", default=None,
                        help="Apply calibrated sim params. Optionally provide a custom path (default: tests/sim2real/config/sim_params.yaml)")
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

    cfg = load_config(cfg_file)

    print(f"Loading policy from {args.policy}")
    policy, is_jit = load_policy(args.policy, cfg)
    print(f"Policy type: {'JIT' if is_jit else 'ActorCritic'}")

    mj_model = mujoco.MjModel.from_xml_path(cfg["asset"]["mujoco_file"])
    # mj_model = mujoco.MjModel.from_xml_path("resources/T1/T1_locomotion_arena.xml")
    mj_model.opt.timestep = cfg["sim"]["dt"]

    # Apply calibrated joint dynamics (damping, armature, frictionloss)
    if args.sim_params:
        if os.path.exists(args.sim_params):
            with open(args.sim_params) as f:
                sim_params = yaml.safe_load(f)
            joint_mapping = get_joint_dof_indices(mj_model)
            apply_sim_params(mj_model, sim_params, joint_mapping)
            print(f"Applied calibrated sim params from: {args.sim_params}")
        else:
            print(f"Warning: sim params file not found: {args.sim_params} (using XML defaults)")

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    
    # Floor friction handling
    ground_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if ground_geom_id >= 0:
        default_friction = mj_model.geom_friction[ground_geom_id].copy()
        if FRICTION_OVERRIDE is not None:
            if isinstance(FRICTION_OVERRIDE, (list, tuple)):
                mj_model.geom_friction[ground_geom_id] = FRICTION_OVERRIDE
            else:
                mj_model.geom_friction[ground_geom_id][0] = FRICTION_OVERRIDE
            print(f"Ground friction [sliding, torsional, rolling]: {mj_model.geom_friction[ground_geom_id]} (overridden)")
        else:
            print(f"Ground friction [sliding, torsional, rolling]: {default_friction} (default)")
    else:
        print("Warning: 'ground' geom not found in MuJoCo model")
    
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
    filter_weight = cfg["normalization"].get("filter_weight", 0.1)
    filtered_lin_vel = np.zeros(3, dtype=np.float32)
    filtered_ang_vel = np.zeros(3, dtype=np.float32)
    
    # Viewer state
    state = {
        "vx": 0.0, "vy": 0.0, "vyaw": 0.0,
        "reset": False,
        "is_playing": False,
        "camera_follow": True,
    }
    cmd_increment = {"vx": 0.1, "vy": 0.1, "vyaw": 0.1}
    cmd_limits = {"vx": (-1.0, 1.0), "vy": (-1.0, 1.0), "vyaw": (-1.0, 1.0)}
    
    # Camera settings from config
    cam_offset = CAM_OFFSET
    cam_distance, cam_azimuth, cam_elevation = offset_to_spherical(cam_offset)
    
    # Store initial pose for reset
    initial_qpos = mj_data.qpos.copy()
    initial_qvel = np.zeros_like(mj_data.qvel)

    def print_status(filtered_vel=None):
        mode = "PLAY" if state["is_playing"] else "PAUSE"
        cam = "FOLLOW" if state["camera_follow"] else "FREE"
        if filtered_vel is not None:
            print(f"\r[{mode}] [{cam}] Cmd: vx={state['vx']:+.2f} vy={state['vy']:+.2f} vyaw={state['vyaw']:+.2f} | Filt: vx={filtered_vel[0]:+.2f} vy={filtered_vel[1]:+.2f} vyaw={filtered_vel[2]:+.2f}    ", end="", flush=True)
        else:
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
            
            # Update filtered velocity at policy rate
            if it % cfg["control"]["decimation"] == 0:
                # Rotate velocities from world to local frame (matches Isaac Gym)
                quat_current = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
                base_lin_vel_local = quat_rotate_inverse(quat_current, mj_data.qvel[0:3])
                base_ang_vel_local = quat_rotate_inverse(quat_current, mj_data.qvel[3:6])
                
                filtered_lin_vel[:] = filter_weight * base_lin_vel_local + (1 - filter_weight) * filtered_lin_vel
                filtered_ang_vel[:] = filter_weight * base_ang_vel_local + (1 - filter_weight) * filtered_ang_vel
                print_status([filtered_lin_vel[0], filtered_lin_vel[1], filtered_ang_vel[2]])
