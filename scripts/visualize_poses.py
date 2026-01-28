"""
Visualize robot poses in Isaac Gym.
Useful for checking if a pose/height causes collision with the floor.
"""

import os
import yaml
import argparse
import numpy as np
from isaacgym import gymapi


def find_config_dir(task_name):
    """Find config directory for task."""
    for root, dirs, files in os.walk("envs"):
        for file in files:
            if file == f"{task_name}.yaml":
                return root
    raise FileNotFoundError(f"Config file '{task_name}.yaml' not found in envs/")


def load_config(task_name):
    """Load task config."""
    config_dir = find_config_dir(task_name)
    cfg_path = os.path.join(config_dir, f"{task_name}.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_poses(task_name):
    """Load poses from poses config file."""
    config_dir = find_config_dir(task_name)
    poses_path = os.path.join(config_dir, f"{task_name}_poses.yaml")
    if not os.path.exists(poses_path):
        print(f"Warning: Poses file not found at {poses_path}, using defaults")
        return {
            "default": {"default": 0.0},
            "t_pose": {"default": 0.0},
        }
    with open(poses_path, "r", encoding="utf-8") as f:
        poses = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(f"Loaded poses from: {poses_path}")
    return poses


def get_pose_dict(poses, pose_name):
    """Get pose dictionary by name."""
    if pose_name not in poses:
        available = list(poses.keys())
        raise ValueError(f"Pose '{pose_name}' not found. Available: {available}")
    print(f"Using pose: {pose_name}")
    return poses[pose_name]


def create_sim(gym):
    """Create Isaac Gym simulation."""
    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = False
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create simulation")
    return sim


def load_robot(gym, sim, cfg):
    """Load robot URDF."""
    asset_file = cfg["asset"]["file"]
    asset_root = os.path.dirname(asset_file)
    asset_file = os.path.basename(asset_file)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.collapse_fixed_joints = cfg["asset"]["collapse_fixed_joints"]
    asset_options.replace_cylinder_with_capsule = cfg["asset"]["replace_cylinder_with_capsule"]
    asset_options.flip_visual_attachments = cfg["asset"]["flip_visual_attachments"]
    print(f"Loading URDF: {asset_file}")
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    return robot_asset


def create_env(gym, sim, robot_asset, height):
    """Create environment with robot."""
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, height)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    robot = gym.create_actor(env, robot_asset, pose, "robot", 0, 1)

    props = gym.get_actor_dof_properties(env, robot)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(400.0)
    props["damping"].fill(40.0)
    gym.set_actor_dof_properties(env, robot, props)

    return env, robot


def apply_pose(gym, env, robot, robot_asset, pose_dict):
    """Apply pose to robot DOFs."""
    num_dofs = gym.get_actor_dof_count(env, robot)
    dof_names = gym.get_asset_dof_names(robot_asset)

    target_positions = np.zeros(num_dofs, dtype=np.float32)
    for i, name in enumerate(dof_names):
        found = False
        for key, value in pose_dict.items():
            if key != "default" and key in name:
                target_positions[i] = value
                found = True
                break
        if not found and "default" in pose_dict:
            target_positions[i] = pose_dict["default"]

    gym.set_actor_dof_position_targets(env, robot, target_positions)
    print(f"Applied pose to {num_dofs} DOFs")
    return target_positions


def setup_viewer(gym, sim, env):
    """Create viewer with camera."""
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer")
    cam_pos = gymapi.Vec3(2.0, -2.0, 1.5)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.7)
    gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)
    return viewer


def main():
    parser = argparse.ArgumentParser(description="Visualize robot poses in Isaac Gym")
    parser.add_argument("--task", type=str, default="T1", help="Task name (default: T1)")
    parser.add_argument("--pose", type=str, default="default", help="Pose name (see t1_poses.yaml)")
    parser.add_argument("--height", type=float, default=None, help="Override height [m]")
    parser.add_argument("--list", action="store_true", help="List available poses and exit")
    args = parser.parse_args()

    cfg = load_config(args.task)
    poses = load_poses(args.task)

    if args.list:
        print(f"\nAvailable poses for {args.task}:")
        for name in poses.keys():
            print(f"  - {name}")
        return

    pose_dict = get_pose_dict(poses, args.pose)
    height = args.height if args.height is not None else cfg["init_state"]["pos"][2]

    gym = gymapi.acquire_gym()
    sim = create_sim(gym)
    robot_asset = load_robot(gym, sim, cfg)
    env, robot = create_env(gym, sim, robot_asset, height)
    apply_pose(gym, env, robot, robot_asset, pose_dict)
    viewer = setup_viewer(gym, sim, env)

    print(f"\n=== Pose Visualization ===")
    print(f"Task: {args.task}")
    print(f"Pose: {args.pose}")
    print(f"Height: {height:.3f} m")
    print("Close window to exit.\n")

    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("Done")


if __name__ == "__main__":
    main()