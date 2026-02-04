#!/usr/bin/env python3
"""
MuJoCo Policy Arena Evaluation
"""
import os
import sys
import shutil
import argparse
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import mujoco

sys.path.insert(0, os.getcwd())
from tests.policies_arena.arena_utils import (
    load_yaml, set_seed, parse_arena_config, get_stage_commands,
    check_survival, compute_tracking_error, check_stage_pass,
    create_output_dir, save_results_csv, print_stage_result,
    print_arena_header, print_arena_summary, quat_rotate_inverse,
    setup_terrain
)
from utils.config_loader import find_experiment_config, load_config
from utils.policy_loader import load_policy

NUM_LEG_JOINTS = 12
LEG_JOINT_START = 0
MIN_BASE_HEIGHT = 0.3


class MuJoCoArenaEvaluator:
    """Evaluates policies through progressive difficulty stages in MuJoCo."""
    
    def __init__(
        self,
        arena_config_path: str,
        task: str = "T1",
        headless_override: Optional[bool] = None,
        num_trials: int = 10,
        sim_params: str = "default"
    ):
        self.arena_config_path = arena_config_path
        self.task = task
        self.num_trials = num_trials
        self.sim_params_mode = sim_params
        
        arena_cfg = load_yaml(arena_config_path)
        self.config = parse_arena_config(arena_cfg, headless_override)
        
        self.headless = self.config['evaluation']['headless']
        self.survival_threshold = self.config['evaluation']['survival_threshold']
        self.base_seed = self.config['evaluation']['seed']
        self.timing = self.config['timing']
        self.terrains = self.config['terrains']
        self.stages = self.config['stages']
        self.policies_config = self.config['policies']
        
        self.output_dir, self.timestamp = create_output_dir(task)
        shutil.copy(arena_config_path, os.path.join(self.output_dir, "arena_config.yaml"))
        
        self.results = []
        self.policy_status = OrderedDict()
        for p in self.policies_config:
            self.policy_status[p['name']] = {
                'alive': True,
                'final_stage': 0,
                'checkpoint': p['checkpoint']
            }
        
        self._load_task_config()
        self._setup_mujoco()
        self._load_sim_params()
        
        print_arena_header(
            self.policies_config, num_trials, len(self.stages),
            self.timing['stage_duration_s'], task, self.output_dir,
            self.headless, sim_params
        )
    
    def _load_task_config(self):
        """Load task configuration."""
        sample_checkpoint = self.policies_config[0]['checkpoint']
        cfg_file = find_experiment_config(sample_checkpoint)
        if not cfg_file:
            cfg_file = os.path.join("envs", "locomotion", f"{self.task}.yaml")
        self.task_cfg = load_config(cfg_file)
        shutil.copy(cfg_file, os.path.join(self.output_dir, "task_config.yaml"))
    
    def _setup_mujoco(self):
        """Setup MuJoCo model and data structures."""
        arena_xml = self.task_cfg["asset"].get("mujoco_arena_file")
        if not arena_xml:
            base_xml = self.task_cfg["asset"]["mujoco_file"]
            arena_xml = base_xml.replace(".xml", "_arena.xml")
        
        if not os.path.exists(arena_xml):
            print(f"[!] Arena XML not found: {arena_xml}")
            print(f"    Falling back to: {self.task_cfg['asset']['mujoco_file']}")
            arena_xml = self.task_cfg["asset"]["mujoco_file"]
        
        print(f"Loading MuJoCo model: {arena_xml}")
        self.mj_model = mujoco.MjModel.from_xml_path(arena_xml)
        self.mj_model.opt.timestep = self.task_cfg["sim"]["dt"]
        self.mj_data = mujoco.MjData(self.mj_model)
        
        self.num_actuators = self.mj_model.nu
        self.default_dof_pos = np.zeros(self.num_actuators, dtype=np.float32)
        self.dof_stiffness = np.zeros(self.num_actuators, dtype=np.float32)
        self.dof_damping = np.zeros(self.num_actuators, dtype=np.float32)
        
        for i in range(self.num_actuators):
            actuator_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            
            found = False
            for name in self.task_cfg["init_state"]["default_joint_angles"].keys():
                if name in actuator_name:
                    self.default_dof_pos[i] = self.task_cfg["init_state"]["default_joint_angles"][name]
                    found = True
                    break
            if not found:
                self.default_dof_pos[i] = self.task_cfg["init_state"]["default_joint_angles"]["default"]
            
            found = False
            for name in self.task_cfg["control"]["stiffness"].keys():
                if name in actuator_name:
                    self.dof_stiffness[i] = self.task_cfg["control"]["stiffness"][name]
                    self.dof_damping[i] = self.task_cfg["control"]["damping"][name]
                    found = True
                    break
            if not found:
                raise ValueError(f"PD gain of joint {actuator_name} not defined")
        
        self.initial_qpos = np.concatenate([
            np.array(self.task_cfg["init_state"]["pos"], dtype=np.float32),
            np.array(self.task_cfg["init_state"]["rot"][3:4] + self.task_cfg["init_state"]["rot"][0:3], dtype=np.float32),
            self.default_dof_pos,
        ])
        self.initial_qvel = np.zeros(self.mj_model.nv, dtype=np.float32)
    
    def _load_sim_params(self):
        """Load simulation parameters (calibrated or default)."""
        if self.sim_params_mode == "calibrated":
            sim_params_path = "tests/sim2real/config/sim_params.yaml"
            if os.path.exists(sim_params_path):
                print(f"Loading calibrated sim params from: {sim_params_path}")
                sim_params = load_yaml(sim_params_path)
                mj_params = sim_params.get('mujoco', {}).get('joint', {})
                
                for i in range(self.num_actuators):
                    actuator_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    joint_id = self.mj_model.actuator_trnid[i, 0]
                    
                    if 'damping' in mj_params:
                        for joint_idx, damping_val in mj_params['damping'].items():
                            if str(joint_idx) in actuator_name or self._joint_matches(joint_idx, actuator_name):
                                self.mj_model.dof_damping[joint_id] = damping_val
                                break
                    
                    if 'armature' in mj_params:
                        for joint_idx, armature_val in mj_params['armature'].items():
                            if str(joint_idx) in actuator_name or self._joint_matches(joint_idx, actuator_name):
                                self.mj_model.dof_armature[joint_id] = armature_val
                                break
                
                print(f"  Applied calibrated damping and armature values")
            else:
                print(f"[!] Calibrated sim params not found: {sim_params_path}, using defaults")
        else:
            print("Using default simulation parameters")
    
    def _joint_matches(self, joint_idx: int, actuator_name: str) -> bool:
        """Check if joint index corresponds to actuator name."""
        joint_map = {
            11: "Left_Hip_Pitch", 12: "Left_Hip_Roll", 13: "Left_Hip_Yaw",
            14: "Left_Knee_Pitch", 15: "Left_Ankle_Pitch", 16: "Left_Ankle_Roll",
            17: "Right_Hip_Pitch", 18: "Right_Hip_Roll", 19: "Right_Hip_Yaw",
            20: "Right_Knee_Pitch", 21: "Right_Ankle_Pitch", 22: "Right_Ankle_Roll",
        }
        expected_name = joint_map.get(joint_idx, "")
        return expected_name in actuator_name
    
    def _reset_robot(self):
        """Reset robot to initial pose."""
        self.mj_data.qpos[:] = self.initial_qpos
        self.mj_data.qvel[:] = self.initial_qvel
        mujoco.mj_forward(self.mj_model, self.mj_data)
    
    def _build_observation(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        quat: np.ndarray,
        base_ang_vel: np.ndarray,
        commands: Tuple[float, float, float],
        gait_process: float,
        gait_frequency: float,
        prev_actions: np.ndarray
    ) -> np.ndarray:
        """Build observation vector matching Isaac Gym format."""
        cfg = self.task_cfg
        num_obs = cfg["env"]["num_observations"]
        obs = np.zeros(num_obs, dtype=np.float32)
        
        projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        lin_vel_x, lin_vel_y, ang_vel_yaw = commands
        
        obs[0:3] = projected_gravity * cfg["normalization"]["gravity"]
        obs[3:6] = base_ang_vel * cfg["normalization"]["ang_vel"]
        obs[6] = lin_vel_x * cfg["normalization"]["lin_vel"] * (gait_frequency > 1.0e-8)
        obs[7] = lin_vel_y * cfg["normalization"]["lin_vel"] * (gait_frequency > 1.0e-8)
        obs[8] = ang_vel_yaw * cfg["normalization"]["ang_vel"] * (gait_frequency > 1.0e-8)
        obs[9] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
        obs[10] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
        obs[11:23] = (dof_pos - self.default_dof_pos) * cfg["normalization"]["dof_pos"]
        obs[23:35] = dof_vel * cfg["normalization"]["dof_vel"]
        obs[35:47] = prev_actions
        
        return obs
    
    def _run_trial(
        self,
        policy,
        is_jit: bool,
        stage: dict,
        terrain_cfg: dict,
        seed: int
    ) -> Tuple[bool, Optional[float], List[float]]:
        """
        Run single trial for a policy on a stage.
        
        Returns: (survived, tracking_error, velocity_history)
        """
        set_seed(seed)
        setup_terrain(self.mj_model, terrain_cfg, seed)
        self._reset_robot()
        
        cfg = self.task_cfg
        dt = cfg["sim"]["dt"]
        decimation = cfg["control"]["decimation"]
        stage_duration = self.timing['stage_duration_s']
        metrics_start = stage_duration * (1 - self.timing['metrics_window_ratio'])
        
        actions = np.zeros(cfg["env"]["num_actions"], dtype=np.float32)
        dof_targets = np.copy(self.default_dof_pos)
        gait_process = 0.0
        
        velocity_errors = []
        survived = True
        t = 0.0
        step = 0
        
        # Velocity filtering (exponential moving average)
        filter_weight = 0.02  # Applied at 50Hz (policy rate), not 500Hz
        filtered_lin_vel = np.zeros(3, dtype=np.float32)
        filtered_ang_vel = np.zeros(3, dtype=np.float32)
        
        while t < stage_duration:
            vx, vy, vyaw = get_stage_commands(stage, t, self.timing)
            
            if vx == 0 and vy == 0 and vyaw == 0:
                gait_frequency = 0.0
            else:
                gait_frequency = np.average(cfg["commands"]["gait_frequency"])
            
            dof_pos = self.mj_data.qpos[7:].astype(np.float32)
            dof_vel = self.mj_data.qvel[6:].astype(np.float32)
            quat = self.mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
            base_ang_vel = self.mj_data.sensor("angular-velocity").data.astype(np.float32)
            base_height = self.mj_data.qpos[2]
            
            if not check_survival(base_height, MIN_BASE_HEIGHT):
                survived = False
                break
            
            if step % decimation == 0:
                obs = self._build_observation(
                    dof_pos, dof_vel, quat, base_ang_vel,
                    (vx, vy, vyaw), gait_process, gait_frequency, actions
                )
                
                obs_tensor = torch.tensor(obs).unsqueeze(0)
                with torch.no_grad():
                    if is_jit:
                        actions[:] = policy(obs_tensor).numpy().squeeze()
                    else:
                        dist = policy.act(obs_tensor)
                        actions[:] = dist.loc.numpy().squeeze()
                
                actions[:] = np.clip(actions, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
                dof_targets = self.default_dof_pos + cfg["control"]["action_scale"] * actions
            
            ctrl = np.clip(
                self.dof_stiffness * (dof_targets - dof_pos) - self.dof_damping * dof_vel,
                self.mj_model.actuator_ctrlrange[:, 0],
                self.mj_model.actuator_ctrlrange[:, 1],
            )
            self.mj_data.ctrl[:] = ctrl
            
            mujoco.mj_step(self.mj_model, self.mj_data)
            
            if step % decimation == 0:
                base_lin_vel_world = self.mj_data.qvel[0:3]
                base_lin_vel_local = quat_rotate_inverse(quat, base_lin_vel_world)
                base_ang_vel_world = self.mj_data.qvel[3:6]
                
                filtered_lin_vel = filter_weight * base_lin_vel_local + (1 - filter_weight) * filtered_lin_vel
                filtered_ang_vel = filter_weight * base_ang_vel_world + (1 - filter_weight) * filtered_ang_vel
            
            if t >= metrics_start:
                vx_err = abs(vx - filtered_lin_vel[0])
                vy_err = abs(vy - filtered_lin_vel[1])
                vyaw_err = abs(vyaw - filtered_ang_vel[2])
                velocity_errors.append([vx_err, vy_err, vyaw_err])
                if step % 500 == 0:
                    print(f"    t={t:.1f}s cmd=({vx:.2f},{vy:.2f},{vyaw:.2f}) filtered=({filtered_lin_vel[0]:.2f},{filtered_lin_vel[1]:.2f},{filtered_ang_vel[2]:.2f})")
            
            gait_process = np.fmod(gait_process + dt * gait_frequency, 1.0)
            t += dt
            step += 1
        
        if velocity_errors:
            errors = np.array(velocity_errors)
            tracking_error = float(np.max([errors[:, 0].mean(), errors[:, 1].mean(), errors[:, 2].mean()]))
        else:
            tracking_error = None
        return survived, tracking_error, velocity_errors
    
    def _evaluate_policy_on_stage(
        self,
        policy_name: str,
        policy,
        is_jit: bool,
        stage_idx: int
    ) -> Tuple[float, Optional[float], bool]:
        """
        Evaluate a policy on a single stage across multiple trials.
        
        Returns: (survival_rate, mean_tracking_error, passed)
        """
        stage = self.stages[stage_idx]
        terrain_name = stage['terrain']
        terrain_cfg = self.terrains[terrain_name]
        tracking_threshold = stage.get('tracking_threshold')
        
        survivals = []
        tracking_errors = []
        
        for trial in range(self.num_trials):
            seed = self.base_seed + stage_idx * 1000 + trial
            survived, tracking_error, _ = self._run_trial(
                policy, is_jit, stage, terrain_cfg, seed
            )
            survivals.append(survived)
            if tracking_error is not None:
                tracking_errors.append(tracking_error)
        
        survival_rate = np.mean(survivals)
        mean_tracking_error = np.mean(tracking_errors) if tracking_errors else None
        
        passed = check_stage_pass(
            survival_rate, self.survival_threshold,
            mean_tracking_error, tracking_threshold
        )
        
        return survival_rate, mean_tracking_error, passed
    
    def run(self):
        """Run full arena evaluation."""
        for stage_idx, stage in enumerate(self.stages):
            print(f"\n--- Stage {stage_idx+1}/{len(self.stages)}: {stage['name']} ---")
            
            any_alive = any(s['alive'] for s in self.policy_status.values())
            if not any_alive:
                print("[!] All policies eliminated. Stopping evaluation.")
                break
            
            for policy_cfg in self.policies_config:
                policy_name = policy_cfg['name']
                
                if not self.policy_status[policy_name]['alive']:
                    continue
                
                policy, is_jit = load_policy(policy_cfg['checkpoint'], self.task_cfg)
                
                survival_rate, tracking_error, passed = self._evaluate_policy_on_stage(
                    policy_name, policy, is_jit, stage_idx
                )
                
                self.results.append({
                    'policy_name': policy_name,
                    'stage': stage_idx + 1,
                    'stage_name': stage['name'],
                    'terrain': stage['terrain'],
                    'survival_rate': survival_rate,
                    'tracking_error': tracking_error if tracking_error else 0.0,
                    'tracking_threshold': stage.get('tracking_threshold', 0.0),
                    'passed': passed,
                })
                
                print_stage_result(
                    stage_idx, stage['name'], policy_name,
                    survival_rate, tracking_error, passed,
                    self.survival_threshold, stage.get('tracking_threshold')
                )
                
                if passed:
                    self.policy_status[policy_name]['final_stage'] = stage_idx + 1
                else:
                    self.policy_status[policy_name]['alive'] = False
        
        save_results_csv(self.results, self.output_dir)
        print_arena_summary(self.policy_status, len(self.stages), self.output_dir)


def main():
    parser = argparse.ArgumentParser(description='MuJoCo Policy Arena Evaluation')
    parser.add_argument('--task', type=str, default='T1', help='Task name')
    parser.add_argument('--config', type=str, default='tests/policies_arena/arena_config.yaml',
                        help='Path to arena configuration file')
    parser.add_argument('--headless', type=str, default=None, choices=['true', 'false'],
                        help='Override headless setting')
    parser.add_argument('--num-trials', type=int, default=10,
                        help='Number of trials per policy per stage')
    parser.add_argument('--sim-params', type=str, default='default',
                        choices=['default', 'calibrated'],
                        help='Simulation parameters: default or calibrated')
    
    args = parser.parse_args()
    
    headless_override = None
    if args.headless is not None:
        headless_override = args.headless.lower() == 'true'
    
    if not os.path.exists(args.config):
        print(f"[!] Config file not found: {args.config}")
        sys.exit(1)
    
    evaluator = MuJoCoArenaEvaluator(
        args.config,
        args.task,
        headless_override,
        args.num_trials,
        args.sim_params
    )
    
    try:
        evaluator.run()
    except KeyboardInterrupt:
        print("\n[!] Evaluation interrupted by user")
        save_results_csv(evaluator.results, evaluator.output_dir)
        print_arena_summary(evaluator.policy_status, len(evaluator.stages), evaluator.output_dir)
        sys.exit(0)


if __name__ == '__main__':
    # Add --visualize flag
    import sys
    if '--visualize' in sys.argv:
        sys.argv.remove('--visualize')
        # Run single visual trial
        from tests.policies_arena.arena_utils import load_yaml, setup_terrain
        from utils.config_loader import find_experiment_config, load_config  
        from utils.policy_loader import load_policy
        import mujoco.viewer
        
        arena_cfg = load_yaml('tests/policies_arena/arena_config.yaml')
        policy_path = arena_cfg['policies'][0]['checkpoint']
        cfg_file = find_experiment_config(policy_path)
        task_cfg = load_config(cfg_file)
        policy, is_jit = load_policy(policy_path, task_cfg)
        
        mj_model = mujoco.MjModel.from_xml_path("resources/T1/T1_locomotion_arena.xml")
        mj_model.opt.timestep = task_cfg["sim"]["dt"]
        mj_data = mujoco.MjData(mj_model)
        
        # Set flat terrain
        mj_model.hfield_data[:] = 0
        
        print("Press Space to start, watch if robot walks straight with vx=0.5")
        # ... rest of play_mujoco logic with vx=0.5 hardcoded
        
        sys.exit(0)
    
    main()