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
from tests.policies_arena.utils.arena_utils import (
    load_yaml, set_seed, parse_arena_config, get_stage_commands,
    check_survival, compute_tracking_error, check_stage_pass,
    create_output_dir, save_results_csv, print_stage_result,
    print_arena_header, print_arena_summary, quat_rotate_inverse
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
        # Store both XML paths for switching between flat and terrain stages
        self.plane_xml = self.task_cfg["asset"]["mujoco_file"]
        self.arena_xml = self.task_cfg["asset"].get("mujoco_arena_file")
        if not self.arena_xml:
            self.arena_xml = self.plane_xml.replace(".xml", "_arena.xml")
        
        if not os.path.exists(self.arena_xml):
            print(f"[!] Arena XML not found: {self.arena_xml}, rough/slope terrain disabled")
            self.arena_xml = None
        
        # Start with plane model
        print(f"Loading MuJoCo model: {self.plane_xml}")
        self.mj_model = mujoco.MjModel.from_xml_path(self.plane_xml)
        self.current_xml = self.plane_xml
        self.mj_model.opt.timestep = self.task_cfg["sim"]["dt"]
        self.mj_data = mujoco.MjData(self.mj_model)
        
        self._setup_actuators()
    
    def _setup_actuators(self):
        """Setup actuator parameters from config."""
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
    
    def _setup_terrain_for_stage(self, terrain_cfg: dict, seed: int):
        """Setup terrain for a stage, switching XML if needed."""
        terrain_type = terrain_cfg.get('type', 'plane')
        needs_heightfield = terrain_type not in ('flat', 'plane')
        
        # Switch XML if needed
        target_xml = self.arena_xml if needs_heightfield else self.plane_xml
        if target_xml and self.current_xml != target_xml:
            print(f"  Switching to: {os.path.basename(target_xml)}")
            self.mj_model = mujoco.MjModel.from_xml_path(target_xml)
            self.mj_model.opt.timestep = self.task_cfg["sim"]["dt"]
            self.mj_data = mujoco.MjData(self.mj_model)
            self.current_xml = target_xml
            self._setup_actuators()
            self._load_sim_params()
        
        # Setup heightfield terrain if using arena model
        if needs_heightfield and self.mj_model.nhfield > 0:
            nrow = self.mj_model.hfield_nrow[0]
            ncol = self.mj_model.hfield_ncol[0]
            max_height = self.mj_model.hfield_size[0][2]
            size_x = self.mj_model.hfield_size[0][0]
            
            # X coordinates for each column
            x = np.linspace(-size_x, size_x, ncol)
            
            if 'random_height' in terrain_cfg:
                # Rough terrain: random bumps
                rng = np.random.RandomState(seed)
                terrain = rng.uniform(0, terrain_cfg['random_height'], (nrow, ncol))
            elif 'slope' in terrain_cfg:
                # Slope terrain: 1.5m flat start, then continuous slope
                FLAT_START = 1.5  # Hardcoded flat start distance
                slope_gradient = terrain_cfg['slope']
                
                heights = np.zeros(ncol)
                for i, xi in enumerate(x):
                    if xi < FLAT_START:
                        heights[i] = 0.0
                    else:
                        heights[i] = slope_gradient * (xi - FLAT_START)
                
                terrain = np.tile(heights, (nrow, 1))
            else:
                terrain = np.zeros((nrow, ncol))
            
            normalized = terrain / max_height if max_height > 0 else terrain
            normalized = np.clip(normalized, 0, 1)
            self.mj_model.hfield_data[:] = normalized.flatten().astype(np.float32)
    
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
    ) -> Tuple[bool, Optional[float], Optional[float], List[float]]:
        """
        Run single trial for a policy on a stage.
        
        Returns: (survived, tracking_error_cmd, tracking_error_full, velocity_history)
        """
        set_seed(seed)
        self._setup_terrain_for_stage(terrain_cfg, seed)
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
        filter_weight = cfg["normalization"].get("filter_weight", 0.1)
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
            
            # Update filtered velocity at policy rate (matches play_mujoco)
            if step % decimation == 0:
                quat_current = self.mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
                base_lin_vel_local = quat_rotate_inverse(quat_current, self.mj_data.qvel[0:3])
                base_ang_vel_local = quat_rotate_inverse(quat_current, self.mj_data.qvel[3:6])
                filtered_lin_vel = filter_weight * base_lin_vel_local + (1 - filter_weight) * filtered_lin_vel
                filtered_ang_vel = filter_weight * base_ang_vel_local + (1 - filter_weight) * filtered_ang_vel
            if t >= metrics_start:
                vx_err = abs(vx - filtered_lin_vel[0])
                vy_err = abs(vy - filtered_lin_vel[1])
                vyaw_err = abs(vyaw - filtered_ang_vel[2])
                velocity_errors.append([vx_err, vy_err, vyaw_err])
            gait_process = np.fmod(gait_process + dt * gait_frequency, 1.0)
            t += dt
            step += 1
        
        if velocity_errors:
            errors = np.array(velocity_errors)
            # Per-dimension mean errors
            err_vx = float(errors[:, 0].mean())
            err_vy = float(errors[:, 1].mean())
            err_vyaw = float(errors[:, 2].mean())
            
            # Full tracking error (all dimensions)
            tracking_error_full = float(np.max([err_vx, err_vy, err_vyaw]))
            
            # Commanded-only tracking error (for rough terrain pass criteria)
            cmd_vx, cmd_vy, cmd_vyaw = stage['vx'], stage['vy'], stage['vyaw']
            commanded_errors = []
            if abs(cmd_vx) > 0.01:
                commanded_errors.append(err_vx)
            if abs(cmd_vy) > 0.01:
                commanded_errors.append(err_vy)
            if abs(cmd_vyaw) > 0.01:
                commanded_errors.append(err_vyaw)
            
            if commanded_errors:
                tracking_error_cmd = float(np.max(commanded_errors))
            else:
                tracking_error_cmd = tracking_error_full
            
            per_dim_errors = (err_vx, err_vy, err_vyaw)
        else:
            tracking_error_cmd = None
            tracking_error_full = None
            per_dim_errors = None
        return survived, tracking_error_cmd, tracking_error_full, per_dim_errors
    
    def _evaluate_policy_on_stage(
        self,
        policy_name: str,
        policy,
        is_jit: bool,
        stage_idx: int
    ) -> Tuple[float, Optional[float], Optional[float], bool]:
        """
        Evaluate a policy on a single stage across multiple trials.
        
        Returns: (survival_rate, mean_tracking_error_cmd, mean_tracking_error_full, passed)
        """
        stage = self.stages[stage_idx]
        terrain_name = stage['terrain']
        terrain_cfg = self.terrains[terrain_name]
        tracking_threshold = stage.get('tracking_threshold')
        
        tracking_mode = terrain_cfg.get('tracking_mode', 'full')
        
        survivals = []
        tracking_errors_cmd = []
        tracking_errors_full = []
        
        per_dim_vx = []
        per_dim_vy = []
        per_dim_vyaw = []
        
        for trial in range(self.num_trials):
            seed = self.base_seed + stage_idx * 1000 + trial
            survived, tracking_error_cmd, tracking_error_full, per_dim = self._run_trial(
                policy, is_jit, stage, terrain_cfg, seed
            )
            survivals.append(survived)
            if tracking_error_cmd is not None:
                tracking_errors_cmd.append(tracking_error_cmd)
            if tracking_error_full is not None:
                tracking_errors_full.append(tracking_error_full)
            if per_dim is not None:
                per_dim_vx.append(per_dim[0])
                per_dim_vy.append(per_dim[1])
                per_dim_vyaw.append(per_dim[2])
        
        survival_rate = np.mean(survivals)
        mean_tracking_error_cmd = np.mean(tracking_errors_cmd) if tracking_errors_cmd else None
        mean_tracking_error_full = np.mean(tracking_errors_full) if tracking_errors_full else None
        mean_per_dim = (
            np.mean(per_dim_vx) if per_dim_vx else 0.0,
            np.mean(per_dim_vy) if per_dim_vy else 0.0,
            np.mean(per_dim_vyaw) if per_dim_vyaw else 0.0
        )
        
        # Use tracking_mode from terrain config: "commanded" or "full"
        if tracking_mode == "commanded":
            pass_tracking_error = mean_tracking_error_cmd
        else:
            pass_tracking_error = mean_tracking_error_full
        
        passed = check_stage_pass(
            survival_rate, self.survival_threshold,
            pass_tracking_error, tracking_threshold
        )
        
        return survival_rate, mean_tracking_error_cmd, mean_tracking_error_full, mean_per_dim, passed
    
    def _get_terrain_group(self, terrain_name: str) -> str:
        """Classify terrain into group: flat, slope, or rough."""
        terrain_cfg = self.terrains[terrain_name]
        if 'random_height' in terrain_cfg:
            return 'rough'
        elif 'slope' in terrain_cfg:
            return 'slope'
        else:
            return 'flat'
    
    def _print_terrain_breakdown(self):
        """Print and save average tracking error per terrain group per policy."""
        if not self.results:
            return
        
        # Group results by policy and terrain group
        policy_terrain_errors = {}
        for r in self.results:
            policy = r['policy_name']
            group = self._get_terrain_group(r['terrain'])
            key = (policy, group)
            if key not in policy_terrain_errors:
                policy_terrain_errors[key] = []
            # Use the display error (cmd for commanded mode, full otherwise)
            terrain_cfg = self.terrains[r['terrain']]
            tracking_mode = terrain_cfg.get('tracking_mode', 'full')
            if tracking_mode == 'commanded':
                policy_terrain_errors[key].append(r['tracking_error_cmd'])
            else:
                policy_terrain_errors[key].append(r['tracking_error_full'])
        
        # Compute averages
        policies = list(dict.fromkeys(r['policy_name'] for r in self.results))
        groups = ['flat', 'slope', 'rough']
        available_groups = [g for g in groups if any((p, g) in policy_terrain_errors for p in policies)]
        
        width = 60
        print(f"\n{'='*width}")
        print("TERRAIN BREAKDOWN (avg tracking error)".center(width))
        print(f"{'='*width}")
        
        # Header
        header = f"  {'Policy':<14}"
        for g in available_groups:
            header += f"{g.capitalize():>10}"
        header += f"{'Overall':>10}"
        print(header)
        print(f"  {'-'*(len(header)-2)}")
        
        # Rows + collect for CSV
        summary_rows = []
        for policy in policies:
            row = f"  {policy:<14}"
            all_errors = []
            row_dict = {'policy_name': policy}
            for g in available_groups:
                key = (policy, g)
                if key in policy_terrain_errors and policy_terrain_errors[key]:
                    avg = np.mean(policy_terrain_errors[key])
                    row += f"{avg:>10.3f}"
                    all_errors.extend(policy_terrain_errors[key])
                    row_dict[f'{g}_avg_error'] = round(avg, 4)
                else:
                    row += f"{'N/A':>10}"
                    row_dict[f'{g}_avg_error'] = None
            
            overall = np.mean(all_errors) if all_errors else 0.0
            row += f"{overall:>10.3f}"
            row_dict['overall_avg_error'] = round(overall, 4)
            print(row)
            summary_rows.append(row_dict)
        
        print(f"{'='*width}")
        
        # Save summary CSV
        save_results_csv(summary_rows, self.output_dir, filename="terrain_summary.csv")
    
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
                
                survival_rate, tracking_error_cmd, tracking_error_full, per_dim, passed = self._evaluate_policy_on_stage(
                    policy_name, policy, is_jit, stage_idx
                )
                
                terrain_cfg = self.terrains[stage['terrain']]
                tracking_mode = terrain_cfg.get('tracking_mode', 'full')
                
                self.results.append({
                    'policy_name': policy_name,
                    'stage': stage_idx + 1,
                    'stage_name': stage['name'],
                    'terrain': stage['terrain'],
                    'survival_rate': survival_rate,
                    'tracking_error_cmd': tracking_error_cmd if tracking_error_cmd else 0.0,
                    'tracking_error_full': tracking_error_full if tracking_error_full else 0.0,
                    'vx_error': per_dim[0],
                    'vy_error': per_dim[1],
                    'vyaw_error': per_dim[2],
                    'tracking_threshold': stage.get('tracking_threshold', 0.0),
                    'passed': passed,
                })
                
                # Print with indicator for rough terrain (commanded-only)
                if tracking_mode == "commanded":
                    display_error = tracking_error_cmd
                    suffix = f" (cmd-only, full={tracking_error_full:.3f})" if tracking_error_full else " (cmd-only)"
                else:
                    display_error = tracking_error_full
                    suffix = f" (full, cmd={tracking_error_cmd:.3f})" if tracking_error_cmd else ""
                
                # Add per-dimension breakdown
                dim_str = f" [vx={per_dim[0]:.3f}, vy={per_dim[1]:.3f}, vyaw={per_dim[2]:.3f}]"
                
                print_stage_result(
                    stage_idx, stage['name'], policy_name,
                    survival_rate, display_error, passed,
                    self.survival_threshold, stage.get('tracking_threshold'),
                    suffix + dim_str
                )
                
                if passed:
                    self.policy_status[policy_name]['final_stage'] = stage_idx + 1
        
        save_results_csv(self.results, self.output_dir)
        self._print_terrain_breakdown()
        print_arena_summary(self.policy_status, len(self.stages), self.output_dir)

def main():
    parser = argparse.ArgumentParser(description='MuJoCo Policy Arena Evaluation')
    parser.add_argument('--task', type=str, default='T1', help='Task name')
    parser.add_argument('--config', type=str, default='tests/policies_arena/config/arena_config.yaml',
                        help='Path to arena configuration file')
    parser.add_argument('--headless', type=str, default=None, choices=['true', 'false'],
                        help='Override headless setting')
    parser.add_argument('--num-trials', type=int, default=32,
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
    main()