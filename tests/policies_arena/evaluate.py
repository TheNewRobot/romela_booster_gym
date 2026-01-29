"""
Policy Arena Evaluation Script

Tests locomotion policies through progressive difficulty stages.
Policies that fail a stage are eliminated; survivors advance.

Usage:
    python tests/policies_arena/evaluate.py --config=tests/policies_arena/arena_config.yaml
    python tests/policies_arena/evaluate.py --config=tests/policies_arena/arena_config.yaml --headless=False
"""
import os
import sys
import yaml
import csv
import shutil
import random
from datetime import datetime
from collections import OrderedDict

import numpy as np

# Must import isaacgym before torch modules that use it
import isaacgym
from isaacgym import gymtorch
import torch

from envs import *
from policies.actor_critic import ActorCritic


def load_yaml(path):
    """Load YAML file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_task_config(task_name="T1"):
    """Find and load task configuration file."""
    for root, dirs, files in os.walk("envs"):
        for file in files:
            if file == f"{task_name}.yaml":
                return os.path.join(root, file)
    raise FileNotFoundError(f"Config file '{task_name}.yaml' not found in envs/")


class PolicyArenaEvaluator:
    """Evaluates policies through progressive difficulty stages."""
    
    def __init__(self, config_path, task="T1", headless_override=None):
        """
        Initialize evaluator.
        
        Args:
            config_path: Path to arena_config.yaml
            task: Task/robot name (e.g., 'T1')
            headless_override: Override headless setting from CLI (True/False/None)
        """
        self.config = load_yaml(config_path)
        self.config_path = config_path
        self.task = task
        
        # Apply headless override if specified
        if headless_override is not None:
            self.config['evaluation']['headless'] = headless_override
        
        # Extract config sections
        self._parse_config()
        
        # Setup output directory
        self._setup_output_dir()
        
        # Print header
        self._print_header()
    
    def _parse_config(self):
        """Parse configuration into instance variables."""
        # Evaluation settings
        eval_cfg = self.config['evaluation']
        self.headless = eval_cfg['headless']
        self.num_envs_per_policy = eval_cfg['num_envs_per_policy']
        self.survival_threshold = eval_cfg['survival_threshold']
        self.base_seed = eval_cfg['seed']
        
        # Timing settings
        timing_cfg = self.config['timing']
        self.stage_duration_s = timing_cfg['stage_duration_s']
        self.settle_duration_s = timing_cfg['settle_duration_s']
        self.ramp_duration_s = timing_cfg['ramp_duration_s']
        self.metrics_window_ratio = timing_cfg['metrics_window_ratio']
        
        # Policies
        self.policies_config = self.config['policies']
        self.num_policies = len(self.policies_config)
        
        # In visualization mode, only use first policy with 1 env
        if not self.headless:
            self.num_policies = 1
            self.num_envs_per_policy = 1
            print("[Visualization mode] Using first policy with 1 environment")
        
        self.total_envs = self.num_policies * self.num_envs_per_policy
        
        # Terrains and stages
        self.terrains = self.config['terrains']
        self.stages = self.config['stages']
        
        # Device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Initialize tracking
        self.results = []
        self.policy_status = OrderedDict()
        for p in self.policies_config[:self.num_policies]:
            self.policy_status[p['name']] = {
                'alive': True,
                'final_stage': 0,
                'checkpoint': p['checkpoint']
            }
    
    def _setup_output_dir(self):
        """Create output directory and save config snapshot."""
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.output_dir = os.path.join("exps", self.task, self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config snapshot for reproducibility
        snapshot_path = os.path.join(self.output_dir, "config.yaml")
        shutil.copy(self.config_path, snapshot_path)
    
    def _print_header(self):
        """Print evaluation header."""
        width = 60
        print(f"\n{'='*width}")
        print("POLICY ARENA EVALUATION".center(width))
        print(f"{'='*width}")
        print(f"  Policies:       {[p['name'] for p in self.policies_config[:self.num_policies]]}")
        print(f"  Envs/policy:    {self.num_envs_per_policy}")
        print(f"  Total envs:     {self.total_envs}")
        print(f"  Stages:         {len(self.stages)}")
        print(f"  Stage duration: {self.stage_duration_s}s")
        print(f"  Device:         {self.device}")
        print(f"  Task:           {self.task}")
        print(f"  Headless:       {self.headless}")
        print(f"  Output:         {self.output_dir}")
        print(f"{'='*width}\n")
    
    def _set_seed(self, stage_idx):
        """Set deterministic seed for a stage."""
        seed = self.base_seed + stage_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _create_env(self, terrain_name):
        """
        Create environment with specified terrain.
        
        Args:
            terrain_name: Key from terrains config
            
        Returns:
            T1 environment instance
        """
        # Load base task config
        task_cfg_path = find_task_config(self.task)
        cfg = load_yaml(task_cfg_path)
        
        # Get terrain settings
        terrain_cfg = self.terrains[terrain_name]
        
        # Apply terrain configuration
        cfg['terrain']['type'] = terrain_cfg['type']
        if terrain_cfg['type'] == 'trimesh':
            if 'slope' in terrain_cfg:
                cfg['terrain']['slope'] = terrain_cfg['slope']
            if 'random_height' in terrain_cfg:
                cfg['terrain']['random_height'] = terrain_cfg['random_height']
            if 'proportions' in terrain_cfg:
                cfg['terrain']['terrain_proportions'] = terrain_cfg['proportions']
        
        # Headless configuration
        cfg['basic']['headless'] = self.headless
        cfg['viewer']['record_video'] = False
        
        # Disable perturbations for deterministic evaluation
        cfg['randomization']['push_interval_s'] = 99999.0
        cfg['randomization']['kick_interval_s'] = 99999.0
        
        # Episode length (longer than stage to prevent auto-reset)
        cfg['rewards']['episode_length_s'] = self.stage_duration_s + 10.0
        
        # Number of environments
        cfg['env']['num_envs'] = self.total_envs
        
        # Play mode reduces some randomization
        cfg['env']['play'] = True
        
        # Create environment
        task_class = eval(self.task)
        env = task_class(cfg)
        
        # Disable render callback for headless
        if self.headless:
            env.render = lambda: None
        
        return env
    
    def _load_policies(self, env):
        """
        Load all policy networks.
        
        Args:
            env: Environment instance (for getting dimensions)
            
        Returns:
            List of policy dicts with model, name, checkpoint, and env_slice
        """
        policies = []
        
        for i, policy_cfg in enumerate(self.policies_config[:self.num_policies]):
            name = policy_cfg['name']
            checkpoint_path = policy_cfg['checkpoint']
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            
            # Create model with matching architecture
            model = ActorCritic(
                num_act=env.num_actions,
                num_obs=env.num_obs,
                num_privileged_obs=env.num_privileged_obs
            ).to(self.device)
            
            # Load weights
            model.load_state_dict(checkpoint['model'], strict=False)
            model.eval()
            
            # Compute env slice for this policy
            start_idx = i * self.num_envs_per_policy
            end_idx = (i + 1) * self.num_envs_per_policy
            
            policies.append({
                'name': name,
                'checkpoint': checkpoint_path,
                'model': model,
                'env_slice': slice(start_idx, end_idx)
            })
            
            print(f"  Loaded '{name}' from {checkpoint_path}")
        
        return policies
    
    def _get_command_at_time(self, t, stage_cfg):
        """
        Get velocity command at time t with settle and ramp phases.
        
        Timeline:
          [0, settle] -> (0, 0, 0)
          [settle, settle+ramp] -> linear interpolation
          [settle+ramp, end] -> (vx, vy, vyaw)
        
        Args:
            t: Current time in seconds
            stage_cfg: Stage configuration dict
            
        Returns:
            Tuple (vx, vy, vyaw)
        """
        target_vx = stage_cfg['vx']
        target_vy = stage_cfg['vy']
        target_vyaw = stage_cfg['vyaw']
        
        if t < self.settle_duration_s:
            # Settle phase: stand still
            return 0.0, 0.0, 0.0
        
        elif t < self.settle_duration_s + self.ramp_duration_s:
            # Ramp phase: linear interpolation to target
            progress = (t - self.settle_duration_s) / self.ramp_duration_s
            return (
                target_vx * progress,
                target_vy * progress,
                target_vyaw * progress
            )
        
        else:
            # Hold phase: full target command
            return target_vx, target_vy, target_vyaw
    
    def _run_stage(self, env, policies, stage_idx, stage_cfg):
        """
        Run a single evaluation stage.
        
        Args:
            env: Environment instance
            policies: List of policy dicts
            stage_idx: Stage index (0-based)
            stage_cfg: Stage configuration
            
        Returns:
            List of result dicts for this stage
        """
        stage_name = stage_cfg['name']
        terrain_name = stage_cfg['terrain']
        tracking_threshold = stage_cfg.get('tracking_threshold')
        
        print(f"\nStage {stage_idx + 1}/{len(self.stages)}: {stage_name}")
        print(f"  Terrain: {terrain_name}, Commands: vx={stage_cfg['vx']}, vy={stage_cfg['vy']}, vyaw={stage_cfg['vyaw']}")
        
        # Set seed for reproducibility
        self._set_seed(stage_idx)
        
        # Reset environment
        obs, _ = env.reset()
        obs = obs.to(self.device)
        
        # Tracking buffers
        ever_terminated = torch.zeros(self.total_envs, dtype=torch.bool, device=self.device)
        
        # Metrics accumulators (collected during metrics window)
        metrics_start_time = self.stage_duration_s * (1.0 - self.metrics_window_ratio)
        vx_errors = []
        vy_errors = []
        vyaw_errors = []
        heights = []
        orientations = []
        powers = []
        
        # Get timing info
        dt = env.dt
        num_steps = int(self.stage_duration_s / dt)
        
        # Simulation loop
        t = 0.0
        for step in range(num_steps):
            # Get current command based on time
            cmd_vx, cmd_vy, cmd_vyaw = self._get_command_at_time(t, stage_cfg)
            
            # Set commands for all environments
            env.commands[:, 0] = cmd_vx
            env.commands[:, 1] = cmd_vy
            env.commands[:, 2] = cmd_vyaw
            
            # Get actions from each policy
            actions = torch.zeros(self.total_envs, env.num_actions, device=self.device)
            
            with torch.no_grad():
                for policy in policies:
                    # Skip eliminated policies
                    if not self.policy_status[policy['name']]['alive']:
                        continue
                    
                    env_slice = policy['env_slice']
                    policy_obs = obs[env_slice]
                    
                    # Get action from policy (use mean, not sampled)
                    dist = policy['model'].act(policy_obs)
                    actions[env_slice] = dist.loc
            
            # Step environment
            obs, rew, dones, infos = env.step(actions)
            obs = obs.to(self.device)
            
            # Track terminations (once terminated, always terminated for this stage)
            ever_terminated |= dones
            
            # Collect metrics during metrics window
            if t >= metrics_start_time:
                # Velocity tracking errors
                vx_err = (env.commands[:, 0] - env.filtered_lin_vel[:, 0]).abs()
                vy_err = (env.commands[:, 1] - env.filtered_lin_vel[:, 1]).abs()
                vyaw_err = (env.commands[:, 2] - env.filtered_ang_vel[:, 2]).abs()
                
                vx_errors.append(vx_err.clone())
                vy_errors.append(vy_err.clone())
                vyaw_errors.append(vyaw_err.clone())
                
                # Base height above terrain
                terrain_h = env.terrain.terrain_heights(env.base_pos)
                heights.append((env.base_pos[:, 2] - terrain_h).clone())
                
                # Orientation error (deviation from upright)
                grav_xy = env.projected_gravity[:, :2]
                orientations.append(torch.norm(grav_xy, dim=-1).clone())
                
                # Power consumption (positive mechanical power only)
                power = (env.torques * env.dof_vel).clamp(min=0).sum(dim=-1)
                powers.append(power.clone())
            
            t += dt
            
            # Render for visualization mode
            if not self.headless:
                env.render()
        
        # Compute metrics per policy
        stage_results = []
        
        for policy in policies:
            policy_name = policy['name']
            
            # Skip eliminated policies
            if not self.policy_status[policy_name]['alive']:
                print(f"  {policy_name}: --SKIPPED-- (eliminated)")
                continue
            
            env_slice = policy['env_slice']
            num_policy_envs = env_slice.stop - env_slice.start
            
            # Survival rate
            policy_terminated = ever_terminated[env_slice]
            num_survived = (~policy_terminated).sum().item()
            survival_rate = num_survived / num_policy_envs
            
            # Compute mean metrics for surviving environments
            surviving_mask = ~policy_terminated
            
            if num_survived > 0:
                # Stack all timesteps and extract this policy's surviving envs
                vx_stack = torch.stack(vx_errors)[:, env_slice]
                vy_stack = torch.stack(vy_errors)[:, env_slice]
                vyaw_stack = torch.stack(vyaw_errors)[:, env_slice]
                h_stack = torch.stack(heights)[:, env_slice]
                o_stack = torch.stack(orientations)[:, env_slice]
                p_stack = torch.stack(powers)[:, env_slice]
                
                # Mean over time and surviving envs
                vx_err_mean = vx_stack[:, surviving_mask].mean().item()
                vy_err_mean = vy_stack[:, surviving_mask].mean().item()
                vyaw_err_mean = vyaw_stack[:, surviving_mask].mean().item()
                height_mean = h_stack[:, surviving_mask].mean().item()
                orient_mean = o_stack[:, surviving_mask].mean().item()
                power_mean = p_stack[:, surviving_mask].mean().item()
            else:
                # All failed - use NaN
                vx_err_mean = float('nan')
                vy_err_mean = float('nan')
                vyaw_err_mean = float('nan')
                height_mean = float('nan')
                orient_mean = float('nan')
                power_mean = float('nan')
            
            # Determine pass/fail
            passed = survival_rate >= self.survival_threshold
            
            if passed and tracking_threshold is not None:
                # Check tracking error (max of all components)
                max_track_err = max(vx_err_mean, vy_err_mean, vyaw_err_mean)
                passed = passed and (max_track_err < tracking_threshold)
            
            # Update policy status
            if passed:
                self.policy_status[policy_name]['final_stage'] = stage_idx + 1
            else:
                self.policy_status[policy_name]['alive'] = False
            
            # Build result record
            result = {
                'policy_name': policy_name,
                'checkpoint': policy['checkpoint'],
                'stage': stage_idx + 1,
                'stage_name': stage_name,
                'terrain': terrain_name,
                'vx_cmd': stage_cfg['vx'],
                'vy_cmd': stage_cfg['vy'],
                'vyaw_cmd': stage_cfg['vyaw'],
                'survival_rate': survival_rate,
                'vx_error': vx_err_mean,
                'vy_error': vy_err_mean,
                'vyaw_error': vyaw_err_mean,
                'mean_height': height_mean,
                'orientation_error': orient_mean,
                'mean_power': power_mean,
                'passed': passed
            }
            
            stage_results.append(result)
            self.results.append(result)
            
            # Print result
            status = "PASS ✓" if passed else "FAIL ✗"
            
            if tracking_threshold is not None:
                print(f"  {policy_name}: {status}")
                print(f"    Survival: {survival_rate*100:.1f}% ({num_survived}/{num_policy_envs})")
                print(f"    Tracking: vx={vx_err_mean:.3f}, vy={vy_err_mean:.3f}, vyaw={vyaw_err_mean:.3f} (threshold: {tracking_threshold})")
            else:
                print(f"  {policy_name}: {status}")
                print(f"    Survival: {survival_rate*100:.1f}% ({num_survived}/{num_policy_envs})")
                print(f"    Height: {height_mean:.3f}m")
            
            if not passed:
                print(f"    → ELIMINATED")
        
        return stage_results
    
    def run(self):
        """Run full evaluation through all stages."""
        # Group stages by terrain to minimize environment recreation
        terrain_to_stages = OrderedDict()
        for idx, stage in enumerate(self.stages):
            terrain = stage['terrain']
            if terrain not in terrain_to_stages:
                terrain_to_stages[terrain] = []
            terrain_to_stages[terrain].append((idx, stage))
        
        # Track which stages we've processed
        processed_stages = set()
        
        # Process stages in order, grouping by terrain
        for stage_idx, stage in enumerate(self.stages):
            # Skip if already processed (due to terrain grouping)
            if stage_idx in processed_stages:
                continue
            
            # Check if any policy is still alive
            any_alive = any(s['alive'] for s in self.policy_status.values())
            if not any_alive:
                print("\n[!] All policies eliminated. Stopping evaluation.")
                break
            
            terrain = stage['terrain']
            
            # Create environment for this terrain
            print(f"\n{'─'*60}")
            print(f"Creating environment for terrain: {terrain}")
            print(f"{'─'*60}")
            
            env = self._create_env(terrain)
            
            # Load all policies
            print("\nLoading policies...")
            policies = self._load_policies(env)
            
            # Run all stages that use this terrain
            for idx, stg in terrain_to_stages[terrain]:
                # Skip if already processed or out of order
                if idx in processed_stages or idx < stage_idx:
                    continue
                
                processed_stages.add(idx)
                
                # Check if any policy is still alive
                any_alive = any(s['alive'] for s in self.policy_status.values())
                if not any_alive:
                    break
                
                # Run the stage
                self._run_stage(env, policies, idx, stg)
            
            # Cleanup environment
            print(f"\nCleaning up environment for terrain: {terrain}")
            del env
            del policies
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save results and print summary
        self._save_results()
        self._print_summary()
    
    def _save_results(self):
        """Save results to CSV file."""
        if not self.results:
            print("\n[!] No results to save.")
            return
        
        csv_path = os.path.join(self.output_dir, "results.csv")
        
        # Get field names from first result
        fieldnames = list(self.results[0].keys())
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"\n[✓] Results saved to: {csv_path}")
    
    def _print_summary(self):
        """Print final summary of results."""
        width = 60
        max_stages = len(self.stages)
        
        print(f"\n{'='*width}")
        print("FINAL RESULTS".center(width))
        print(f"{'='*width}")
        
        for policy_name, status in self.policy_status.items():
            final_stage = status['final_stage']
            
            if final_stage == max_stages:
                marker = "★ CHAMPION"
            elif final_stage == 0:
                marker = "(failed stage 1)"
            else:
                marker = ""
            
            print(f"  {policy_name}: Stage {final_stage}/{max_stages} {marker}")
        
        print(f"{'='*width}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Results CSV:      {os.path.join(self.output_dir, 'results.csv')}")
        print(f"  Config snapshot:  {os.path.join(self.output_dir, 'config.yaml')}")
        print(f"{'='*width}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Policy Arena Evaluation - Test locomotion policies through progressive difficulty stages'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='T1',
        help='Task/robot name (default: T1)'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='tests/policies_arena/arena_config.yaml',
        help='Path to arena configuration file'
    )
    parser.add_argument(
        '--headless',
        type=str,
        default=None,
        choices=['true', 'false', 'True', 'False'],
        help='Override headless setting (true/false)'
    )
    
    args = parser.parse_args()
    
    # Parse headless override
    headless_override = None
    if args.headless is not None:
        headless_override = args.headless.lower() == 'true'
    
    # Check config file exists
    if not os.path.exists(args.config):
        print(f"[!] Config file not found: {args.config}")
        sys.exit(1)
    
    # Run evaluation
    evaluator = PolicyArenaEvaluator(args.config, args.task, headless_override)
    
    try:
        evaluator.run()
    except KeyboardInterrupt:
        print("\n[!] Evaluation interrupted by user")
        evaluator._save_results()
        evaluator._print_summary()
        sys.exit(0)


if __name__ == '__main__':
    main()