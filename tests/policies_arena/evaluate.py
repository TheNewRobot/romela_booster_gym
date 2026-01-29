"""
Policy Arena Evaluation Script

Tests locomotion policies through progressive difficulty stages.
Policies that fail a stage are eliminated; survivors advance.

Usage:
    python tests/policies_arena/evaluate.py --task=T1
    python tests/policies_arena/evaluate.py --task=T1 --config=tests/policies_arena/arena_config.yaml
    python tests/policies_arena/evaluate.py --task=T1 --headless=False

Note: Due to Isaac Gym limitations (one simulation per process), this script
spawns subprocesses for each terrain type automatically.
"""
import os
import sys
import yaml
import csv
import json
import shutil
import random
import subprocess
import tempfile
from datetime import datetime
from collections import OrderedDict

import numpy as np

# Isaac Gym imports (must be before other gym-related imports)
import isaacgym
from envs import *
from policies.actor_critic import ActorCritic
import torch

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
    
    def __init__(self, config_path, task="T1", headless_override=None, worker_mode=False):
        """
        Initialize evaluator.
        
        Args:
            config_path: Path to arena_config.yaml
            task: Task/robot name (e.g., 'T1')
            headless_override: Override headless setting from CLI (True/False/None)
            worker_mode: If True, skip output dir setup (worker will use temp dir)
        """
        self.config = load_yaml(config_path)
        self.config_path = config_path
        self.task = task
        self.worker_mode = worker_mode
        
        # Apply headless override if specified
        if headless_override is not None:
            self.config['evaluation']['headless'] = headless_override
        
        # Extract config sections
        self._parse_config()
        
        # Setup output directory (coordinator only)
        if not worker_mode:
            self._setup_output_dir()
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
        print(f"  Task:           {self.task}")
        print(f"  Device:         {self.device}")
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
            Environment instance
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
        
        # Create environment (dynamically get task class)
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
        
        print(f"\n{'─'*50}")
        print(f"Stage {stage_idx + 1}/{len(self.stages)}: {stage_name}")
        print(f"Terrain: {terrain_name} | Commands: vx={stage_cfg['vx']}, vy={stage_cfg['vy']}, vyaw={stage_cfg['vyaw']}")
        print(f"{'─'*50}")
        
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
                print(f"\n  ► {policy_name}: SKIPPED (eliminated)")
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
            
            print(f"\n  ► {policy_name}: {status}")
            print(f"      Survival: {survival_rate*100:.1f}% ({num_survived}/{num_policy_envs})")
            
            if tracking_threshold is not None:
                print(f"      Tracking: vx={vx_err_mean:.3f}, vy={vy_err_mean:.3f}, vyaw={vyaw_err_mean:.3f}")
            else:
                print(f"      Height: {height_mean:.3f}m")
            
            if not passed:
                print(f"      → ELIMINATED")
        
        return stage_results
    
    def run_worker(self, terrain_name, stage_indices, temp_output_path):
        """
        Worker mode: Run stages for a single terrain and save results.
        
        This is called in a subprocess to work around Isaac Gym's
        one-simulation-per-process limitation.
        
        Args:
            terrain_name: Terrain to use for all stages
            stage_indices: List of stage indices to run
            temp_output_path: Path to write JSON results
        """
        print(f"\n{'─'*60}")
        print(f"[Worker] Creating environment for terrain: {terrain_name}")
        print(f"[Worker] Stages to run: {[i+1 for i in stage_indices]}")
        print(f"{'─'*60}")
        
        # Create environment (only once per worker)
        env = self._create_env(terrain_name)
        
        # Load policies
        print("\n[Worker] Loading policies...")
        policies = self._load_policies(env)
        
        # Run each stage
        for stage_idx in stage_indices:
            # Check if any policy is still alive
            any_alive = any(s['alive'] for s in self.policy_status.values())
            if not any_alive:
                print("\n[Worker] All policies eliminated. Stopping.")
                break
            
            stage_cfg = self.stages[stage_idx]
            self._run_stage(env, policies, stage_idx, stage_cfg)
        
        # Save results and policy status to temp file
        output_data = {
            'results': self.results,
            'policy_status': dict(self.policy_status)
        }
        
        with open(temp_output_path, 'w') as f:
            json.dump(output_data, f)
        
        print(f"\n[Worker] Results saved to: {temp_output_path}")
    
    def run(self):
        """
        Run full evaluation through all stages.
        
        Uses subprocesses for each terrain type to work around
        Isaac Gym's one-simulation-per-process limitation.
        """
        # Group stages by terrain
        terrain_to_stages = OrderedDict()
        for idx, stage in enumerate(self.stages):
            terrain = stage['terrain']
            if terrain not in terrain_to_stages:
                terrain_to_stages[terrain] = []
            terrain_to_stages[terrain].append(idx)
        
        print(f"Terrain groups: {dict(terrain_to_stages)}")
        
        # Create temp directory for inter-process communication
        temp_dir = tempfile.mkdtemp(prefix="arena_eval_")
        print(f"Temp directory: {temp_dir}")
        
        try:
            # Process each terrain group in a subprocess
            for terrain_name, stage_indices in terrain_to_stages.items():
                # Check if any policy is still alive
                any_alive = any(s['alive'] for s in self.policy_status.values())
                if not any_alive:
                    print("\n[!] All policies eliminated. Stopping evaluation.")
                    break
                
                # Prepare temp output path for this worker
                temp_output = os.path.join(temp_dir, f"{terrain_name}.json")
                
                # Build subprocess command
                cmd = [
                    sys.executable,
                    '-u',
                    __file__,
                    f"--task={self.task}",
                    f"--config={self.config_path}",
                    f"--headless={'true' if self.headless else 'false'}",
                    "--worker",
                    f"--terrain={terrain_name}",
                    f"--stages={','.join(str(i) for i in stage_indices)}",
                    f"--temp-output={temp_output}",
                    f"--policy-status={json.dumps(dict(self.policy_status))}"
                ]
                
                print(f"\n{'='*60}")
                print(f"Spawning worker for terrain: {terrain_name}")
                print(f"Stages: {[i+1 for i in stage_indices]}")
                print(f"{'='*60}")
                
                # Run subprocess
                skip_patterns = [
                    'Importing module', 'Setting GYM_USD', 'PyTorch version',
                    'Device count', 'gymtorch', 'Using /home', 'Emitting ninja',
                    'Building extension', 'Allowing ninja', 'ninja:', 'Loading extension',
                    'TypedStorage is deprecated', 'return self.fget', '_bindings/src'
                ]
                
                process = subprocess.Popen(
                    cmd,
                    cwd=os.getcwd(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1  # Line buffered
                )
                
                # Stream output line by line, filtering noise
                for line in process.stdout:
                    line = line.rstrip('\n')
                    if any(skip in line for skip in skip_patterns):
                        continue
                    if line.strip():
                        print(line)
                
                process.wait()
                
                if process.returncode != 0:
                    print(f"\n[!] Worker failed with return code {process.returncode}")
                    break
                
                # Load results from worker
                if os.path.exists(temp_output):
                    with open(temp_output, 'r') as f:
                        worker_data = json.load(f)
                    
                    # Merge results
                    self.results.extend(worker_data['results'])
                    
                    # Update policy status
                    for name, status in worker_data['policy_status'].items():
                        self.policy_status[name] = status
                else:
                    print(f"\n[!] Worker output not found: {temp_output}")
                    break
        
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Save final results and print summary
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
    
    # Worker mode arguments (used internally for subprocesses)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--terrain', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--stages', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--temp-output', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--policy-status', type=str, help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Parse headless override
    headless_override = None
    if args.headless is not None:
        headless_override = args.headless.lower() == 'true'
    
    # Check config file exists
    if not os.path.exists(args.config):
        print(f"[!] Config file not found: {args.config}")
        sys.exit(1)
    
    # Worker mode - run in subprocess for single terrain
    if args.worker:
        evaluator = PolicyArenaEvaluator(
            args.config, 
            args.task, 
            headless_override, 
            worker_mode=True
        )
        
        # Restore policy status from coordinator
        if args.policy_status:
            evaluator.policy_status = OrderedDict(json.loads(args.policy_status))
        
        # Parse stage indices
        stage_indices = [int(i) for i in args.stages.split(',')]
        
        # Run worker
        evaluator.run_worker(args.terrain, stage_indices, args.temp_output)
        return
    
    # Coordinator mode - orchestrate evaluation
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