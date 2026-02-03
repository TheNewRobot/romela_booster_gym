"""
System Identification Joint Test Script

Sends known commands (steps, sinusoids) to individual joints on a hanging robot.
Records joint responses for sim2real parameter tuning.

Usage:
    python sysid_joint_test.py --config=T1.yaml --joint=11

Startup sequence:
    1. Put robot in PREP mode
    2. Hang robot securely
    3. Run this script
    4. Press 'b' to enter custom mode (robot holds position)
    5. Press 'r' to start tests (robot begins moving)
    6. Ctrl+C or PREP button for emergency stop
"""

import numpy as np
import time
import yaml
import logging
import threading
import csv
import os
import argparse
import signal
import sys
from datetime import datetime

from booster_robotics_sdk_python import (
    ChannelFactory,
    B1LocoClient,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    LowCmd,
    LowState,
    B1JointCnt,
    RobotMode,
    LowCmdType,
    MotorCmd,
)

from deploy.utils.remote_control_service import RemoteControlService
from deploy.utils.timer import TimerConfig, Timer


class SysIDController:
    def __init__(self, robot_cfg_file, sysid_cfg_file, test_joint=None):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load robot config (T1.yaml)
        with open(robot_cfg_file, "r") as f:
            self.robot_cfg = yaml.safe_load(f)

        # Load sysid config
        with open(sysid_cfg_file, "r") as f:
            self.sysid_cfg = yaml.safe_load(f)

        # Override joint if specified
        self.test_joint = test_joint if test_joint is not None else self.sysid_cfg["test"]["joint"]
        
        # Get joint name for logging
        self.joint_name = self.sysid_cfg["joint_names"].get(self.test_joint, f"Joint_{self.test_joint}")
        
        # Initialize components
        self.remoteControlService = RemoteControlService()
        self._init_timer()
        self._init_state_values()
        self._init_communication()
        
        self.running = True
        self.test_running = False
        self.data_records = []
        
        # Get neutral position for test joint
        self.neutral_pos = self.robot_cfg["common"]["default_qpos"][self.test_joint]
        
        # Build parallel mechanism pair mapping from config
        # Modify mech.parallel_mech_indexes in robot config if hardware changes
        self.parallel_pairs = {}
        parallel_indexes = self.robot_cfg.get("mech", {}).get("parallel_mech_indexes", [])
        for i in range(0, len(parallel_indexes) - 1, 2):
            j1, j2 = parallel_indexes[i], parallel_indexes[i + 1]
            self.parallel_pairs[j1] = j2
            self.parallel_pairs[j2] = j1
        
        self.logger.info(f"SysID Controller initialized for {self.joint_name} (index {self.test_joint})")
        self.logger.info(f"Neutral position: {self.neutral_pos:.4f} rad")

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=self.robot_cfg["common"]["dt"]))
        self.next_publish_time = self.timer.get_time()

    def _init_state_values(self):
        self.dof_pos = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_vel = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_torque = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_pos_latest = np.zeros(B1JointCnt, dtype=np.float32)
        self.current_cmd = np.zeros(B1JointCnt, dtype=np.float32)

    def _init_communication(self):
        try:
            self.low_cmd = LowCmd()
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.low_cmd_publisher = B1LowCmdPublisher()
            self.client = B1LocoClient()

            self.low_state_subscriber.InitChannel()
            self.low_cmd_publisher.InitChannel()
            self.client.Init()
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _low_state_handler(self, low_state_msg: LowState):
        """Process incoming sensor data - NO IMU check (robot is hanging)"""
        self.timer.tick_timer_if_sim()
        
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos_latest[i] = motor.q
            self.dof_pos[i] = motor.q
            self.dof_vel[i] = motor.dq
            self.dof_torque[i] = motor.tau_est
        
        # Safety: check joint velocity
        max_vel = self.sysid_cfg["safety"]["max_joint_velocity"]
        if abs(self.dof_vel[self.test_joint]) > max_vel:
            self.logger.warning(f"Joint velocity limit exceeded: {self.dof_vel[self.test_joint]:.2f} rad/s")

    def _init_low_cmd(self):
        """Initialize command structure"""
        self.low_cmd.cmd_type = LowCmdType.SERIAL
        self.low_cmd.motor_cmd = [MotorCmd() for _ in range(B1JointCnt)]
        
        for i in range(B1JointCnt):
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].weight = 0.0

    def _setup_joint_gains(self):
        """Set gains: active joint uses common gains, others hang loose.
        
        For parallel mechanism joints (ankles), both paired motors are activated
        to enable proper Jacobian-based series-parallel conversion in the SDK.
        """
        self._init_low_cmd()
        
        passive_kp = self.sysid_cfg["passive_joints"]["kp"]
        passive_kd = self.sysid_cfg["passive_joints"]["kd"]
        
        # Determine which joints need active gains
        active_joints = {self.test_joint}
        paired_joint = self.parallel_pairs.get(self.test_joint)
        if paired_joint is not None:
            active_joints.add(paired_joint)
            paired_name = self.sysid_cfg["joint_names"].get(paired_joint, f"Joint_{paired_joint}")
            self.logger.info(f"Parallel mechanism: also activating {paired_name} (index {paired_joint})")
        
        for i in range(B1JointCnt):
            if i in active_joints:
                # Active joint: use common gains
                if self.sysid_cfg["active_joint"]["use_common_gains"]:
                    self.low_cmd.motor_cmd[i].kp = self.robot_cfg["common"]["stiffness"][i]
                    self.low_cmd.motor_cmd[i].kd = self.robot_cfg["common"]["damping"][i]
                else:
                    self.low_cmd.motor_cmd[i].kp = self.sysid_cfg["active_joint"]["kp"]
                    self.low_cmd.motor_cmd[i].kd = self.sysid_cfg["active_joint"]["kd"]
            else:
                # Passive joint: hang loose
                self.low_cmd.motor_cmd[i].kp = passive_kp
                self.low_cmd.motor_cmd[i].kd = passive_kd
            
            # Set neutral position
            self.low_cmd.motor_cmd[i].q = self.robot_cfg["common"]["default_qpos"][i]
            self.current_cmd[i] = self.robot_cfg["common"]["default_qpos"][i]

    def _send_cmd(self):
        """Send current command to robot"""
        self.low_cmd_publisher.Write(self.low_cmd)

    def _record_data(self, test_type, test_param):
        """Record current state to data buffer"""
        self.data_records.append({
            "timestamp": time.time(),
            "test_type": test_type,
            "test_param": test_param,
            "cmd_position": self.current_cmd[self.test_joint],
            "actual_position": self.dof_pos[self.test_joint],
            "actual_velocity": self.dof_vel[self.test_joint],
            "actual_torque": self.dof_torque[self.test_joint],
        })

    def _run_step_test(self, amplitude):
        """Run single step response test"""
        # Clamp amplitude for safety
        max_amp = self.sysid_cfg["safety"]["max_amplitude"]
        amplitude = np.clip(amplitude, -max_amp, max_amp)
        
        target = self.neutral_pos + amplitude
        duration = self.sysid_cfg["test"]["step"]["duration"]
        dt = self.robot_cfg["common"]["dt"]
        
        self.logger.info(f"  Step test: {amplitude:+.3f} rad for {duration}s")
        
        # Apply step
        self.low_cmd.motor_cmd[self.test_joint].q = target
        self.current_cmd[self.test_joint] = target
        
        start_time = time.time()
        while time.time() - start_time < duration and self.running:
            self._send_cmd()
            self._record_data("step", amplitude)
            time.sleep(dt)

    def _run_sine_test(self, frequency):
        """Run single sinusoid test"""
        amplitude = self.sysid_cfg["test"]["sine"]["amplitude"]
        max_amp = self.sysid_cfg["safety"]["max_amplitude"]
        amplitude = np.clip(amplitude, -max_amp, max_amp)
        
        duration = self.sysid_cfg["test"]["sine"]["duration"]
        dt = self.robot_cfg["common"]["dt"]
        
        self.logger.info(f"  Sine test: {frequency} Hz, ±{amplitude:.3f} rad for {duration}s")
        
        start_time = time.time()
        while time.time() - start_time < duration and self.running:
            t = time.time() - start_time
            target = self.neutral_pos + amplitude * np.sin(2 * np.pi * frequency * t)
            
            self.low_cmd.motor_cmd[self.test_joint].q = target
            self.current_cmd[self.test_joint] = target
            
            self._send_cmd()
            self._record_data("sine", frequency)
            time.sleep(dt)

    def _return_to_neutral(self):
        """Smoothly return to neutral position"""
        settle_time = self.sysid_cfg["test"]["step"]["settle_time"]
        dt = self.robot_cfg["common"]["dt"]
        
        self.low_cmd.motor_cmd[self.test_joint].q = self.neutral_pos
        self.current_cmd[self.test_joint] = self.neutral_pos
        
        start_time = time.time()
        while time.time() - start_time < settle_time and self.running:
            self._send_cmd()
            time.sleep(dt)

    def run_test_sequence(self):
        """Run full test sequence"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Starting test sequence for {self.joint_name}")
        self.logger.info(f"{'='*50}\n")
        
        # Step tests
        self.logger.info("Running step response tests...")
        for amp in self.sysid_cfg["test"]["step"]["amplitudes"]:
            if not self.running:
                break
            self._run_step_test(amp)
            self._return_to_neutral()
        
        # Sine tests
        self.logger.info("\nRunning sinusoid tests...")
        for freq in self.sysid_cfg["test"]["sine"]["frequencies"]:
            if not self.running:
                break
            self._run_sine_test(freq)
            self._return_to_neutral()
        
        self.logger.info("\nTest sequence complete!")

    def save_data(self, output_dir):
        """Save recorded data to CSV with metadata header"""
        # Get experiment name from config
        experiment_name = self.sysid_cfg.get("experiment", {}).get("name", "unnamed")
        
        # Create experiment subfolder
        experiment_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"joint_{self.test_joint}_{self.joint_name}_{timestamp}.csv"
        filepath = os.path.join(experiment_dir, filename)
        
        with open(filepath, "w", newline="") as f:
            # Write metadata header
            f.write(f"# experiment: {experiment_name}\n")
            f.write(f"# joint_index: {self.test_joint}\n")
            f.write(f"# joint_name: {self.joint_name}\n")
            
            # Write data
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "test_type", "test_param",
                "cmd_position", "actual_position", "actual_velocity", "actual_torque"
            ])
            writer.writeheader()
            writer.writerows(self.data_records)
        
        self.logger.info(f"Data saved to {filepath}")
        self.logger.info(f"Total records: {len(self.data_records)}")
        return filepath

    def start_custom_mode(self):
        """Enter custom mode (same pattern as deploy.py)"""
        print(f"\n{self.remoteControlService.get_custom_mode_operation_hint()}")
        while self.running:
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)
        
        if not self.running:
            return False
        
        # Setup gains and send initial command
        self._setup_joint_gains()
        self._send_cmd()
        
        # Enter custom mode
        self.client.ChangeMode(RobotMode.kCustom)
        self.logger.info("Custom mode activated - robot holding position")
        return True

    def wait_for_test_start(self):
        """Wait for user to start tests"""
        print(f"\n{self.remoteControlService.get_rl_gait_operation_hint().replace('rl Gait', 'test sequence')}")
        while self.running:
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)
        
        return self.running

    def cleanup(self):
        """Safe shutdown"""
        self.running = False
        self.logger.info("Switching to damping mode...")
        try:
            self.client.ChangeMode(RobotMode.kDamping)
        except:
            pass
        
        self.remoteControlService.close()
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Joint System Identification Test")
    parser.add_argument("--config", required=True, type=str, help="Robot config file (e.g., T1.yaml)")
    parser.add_argument("--sysid-config", type=str, default="tests/sim2real/config/sysid_joint_dynamics.yaml",
                        help="SysID test config file")
    parser.add_argument("--joint", type=int, default=None, help="Joint index to test (overrides config)")
    parser.add_argument("--output", type=str, default="tests/sim2real/data", help="Output directory for CSV")
    parser.add_argument("--net", type=str, default="192.168.10.102", help="Network interface for SDK")
    args = parser.parse_args()

    # Resolve config path
    robot_cfg_file = os.path.join("deploy", "configs", args.config)
    
    # Signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("\n\nShutting down safely...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    print(f"\n{'='*50}")
    print("Joint System Identification Test")
    print(f"{'='*50}")
    print(f"Robot config: {robot_cfg_file}")
    print(f"SysID config: {args.sysid_config}")
    print(f"Output dir: {args.output}")
    print(f"Network: {args.net}")
    print(f"{'='*50}\n")

    # Initialize SDK
    print(f"Connecting to robot at {args.net}...")
    ChannelFactory.Instance().Init(0, args.net)

    with SysIDController(robot_cfg_file, args.sysid_config, args.joint) as controller:
        time.sleep(2)  # Wait for channels
        print("Initialization complete.\n")

        # Step 1: Enter custom mode
        if not controller.start_custom_mode():
            print("Aborted.")
            sys.exit(0)

        # Step 2: Wait for test start
        if not controller.wait_for_test_start():
            print("Aborted.")
            sys.exit(0)

        # Step 3: Run tests
        try:
            controller.run_test_sequence()
        except KeyboardInterrupt:
            print("\nTest interrupted by user.")
        
        # Step 4: Save data
        controller.save_data(args.output)
        
        # Step 5: Return to damping mode
        controller.client.ChangeMode(RobotMode.kDamping)
        print("\nRobot in damping mode. Safe to switch to PREP.")


if __name__ == "__main__":
    main()