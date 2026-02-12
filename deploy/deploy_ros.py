import numpy as np
import time
import yaml
import logging
import threading
import os
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
)

from utils.command import create_prepare_cmd, create_first_frame_rl_cmd
from utils.remote_control_service import RemoteControlService
from utils.rotate import rotate_vector_inverse_rpy
from utils.timer import TimerConfig, Timer
from utils.policy import Policy
from utils.data_logger import DataLogger

ENABLE_ROS2_LOGGING = False

if ENABLE_ROS2_LOGGING:
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import Float32MultiArray
    from booster_msgs.msg import BoosterControlCmd, BoosterMotorCmd


class Controller:
    def __init__(self, cfg_file, task_name="experiment") -> None:
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.remoteControlService = RemoteControlService()
        self.policy = Policy(cfg=self.cfg)

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()
        self.publish_runner = None
        self.running = True

        self.publish_lock = threading.Lock()

        # CSV data logger
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "data", task_name, f"deploy_log_obs_{timestamp}")
        policy_name = os.path.basename(self.cfg["policy"]["policy_path"])
        self.data_logger = DataLogger(output_dir, num_joints=B1JointCnt, policy_name=policy_name)
        self.logger.info(f"Policy: {policy_name}")
        self.logger.info(f"Data logger output: {output_dir}")

        if ENABLE_ROS2_LOGGING:
            self._init_ros2_logging()

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=self.cfg["common"]["dt"]))
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()

    def _init_low_state_values(self):
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_vel = np.zeros(B1JointCnt, dtype=np.float32)

        self.dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.filtered_dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_pos_latest = np.zeros(B1JointCnt, dtype=np.float32)

        # Logging buffers
        self.dof_tau_est = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_temperature = np.zeros(B1JointCnt, dtype=np.float32)
        self.imu_rpy = np.zeros(3, dtype=np.float32)
        self.imu_gyro = np.zeros(3, dtype=np.float32)
        self.imu_acc = np.zeros(3, dtype=np.float32)

    def _init_communication(self) -> None:
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

    def _init_ros2_logging(self):
        rclpy.init()
        self.ros_node = rclpy.create_node('deploy_logger')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )

        self.pub_control_cmd = self.ros_node.create_publisher(BoosterControlCmd, '/booster/control_cmd', qos)
        self.pub_motor_cmd = self.ros_node.create_publisher(BoosterMotorCmd, '/booster/motor_cmd', qos)
        self.pub_policy_obs = self.ros_node.create_publisher(Float32MultiArray, '/booster/policy_obs', qos)
        self.pub_policy_actions = self.ros_node.create_publisher(Float32MultiArray, '/booster/policy_actions', qos)

        self.ctrl_msg = BoosterControlCmd()
        self.obs_msg = Float32MultiArray()
        self.act_msg = Float32MultiArray()
        self.motor_msg = BoosterMotorCmd()

        self.logger.info("ROS2 logging initialized")

    def _low_state_handler(self, low_state_msg: LowState):
        if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
            self.logger.warning("IMU base rpy values are too large: {}".format(low_state_msg.imu_state.rpy))
            self.running = False
        self.timer.tick_timer_if_sim()
        time_now = self.timer.get_time()

        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos_latest[i] = motor.q
            self.dof_tau_est[i] = motor.tau_est
            self.dof_temperature[i] = motor.temperature

        self.imu_rpy[:] = low_state_msg.imu_state.rpy
        self.imu_gyro[:] = low_state_msg.imu_state.gyro
        self.imu_acc[:] = low_state_msg.imu_state.acc

        if time_now >= self.next_inference_time:
            self.projected_gravity[:] = rotate_vector_inverse_rpy(
                low_state_msg.imu_state.rpy[0],
                low_state_msg.imu_state.rpy[1],
                low_state_msg.imu_state.rpy[2],
                np.array([0.0, 0.0, -1.0]),
            )
            self.base_ang_vel[:] = low_state_msg.imu_state.gyro
            for i, motor in enumerate(low_state_msg.motor_state_serial):
                self.dof_pos[i] = motor.q
                self.dof_vel[i] = motor.dq

    def _send_cmd(self, cmd: LowCmd):
        self.low_cmd_publisher.Write(cmd)

    def cleanup(self) -> None:
        self.data_logger.save()
        self.remoteControlService.close()
        if ENABLE_ROS2_LOGGING:
            if rclpy.ok():
                self.ros_node.destroy_node()
                rclpy.shutdown()
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()
        if hasattr(self, "publish_runner") and getattr(self, "publish_runner") != None:
            self.publish_runner.join(timeout=1.0)

    def start_custom_mode_conditionally(self):
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while True:
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)
        start_time = time.perf_counter()
        create_prepare_cmd(self.low_cmd, self.cfg)
        for i in range(B1JointCnt):
            self.dof_target[i] = self.low_cmd.motor_cmd[i].q
            self.filtered_dof_target[i] = self.low_cmd.motor_cmd[i].q
        self._send_cmd(self.low_cmd)
        send_time = time.perf_counter()
        self.logger.debug(f"Send cmd took {(send_time - start_time)*1000:.4f} ms")
        self.client.ChangeMode(RobotMode.kCustom)
        end_time = time.perf_counter()
        self.logger.debug(f"Change mode took {(end_time - send_time)*1000:.4f} ms")

    def start_rl_gait_conditionally(self):
        print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
        while True:
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)
        create_first_frame_rl_cmd(self.low_cmd, self.cfg)
        self._send_cmd(self.low_cmd)
        self.next_inference_time = self.timer.get_time()
        self.next_publish_time = self.timer.get_time()
        self.publish_runner = threading.Thread(target=self._publish_cmd)
        self.publish_runner.daemon = True
        self.publish_runner.start()
        print(f"{self.remoteControlService.get_operation_hint()}")

    def run(self):
        time_now = self.timer.get_time()
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        self.logger.debug("-----------------------------------------------------")
        self.next_inference_time += self.policy.get_policy_interval()
        self.logger.debug(f"Next start time: {self.next_inference_time}")
        start_time = time.perf_counter()

        self.dof_target[:] = self.policy.inference(
            time_now=time_now,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            vx=self.remoteControlService.get_vx_cmd(),
            vy=self.remoteControlService.get_vy_cmd(),
            vyaw=self.remoteControlService.get_vyaw_cmd(),
        )

        inference_time = time.perf_counter()
        self.logger.debug(f"Inference took {(inference_time - start_time)*1000:.4f} ms")

        # CSV logging — single row, flushed to disk
        self.data_logger.log(
            vx=self.remoteControlService.get_vx_cmd(),
            vy=self.remoteControlService.get_vy_cmd(),
            vyaw=self.remoteControlService.get_vyaw_cmd(),
            imu_rpy=self.imu_rpy,
            imu_gyro=self.imu_gyro,
            imu_acc=self.imu_acc,
            dof_pos=self.dof_pos_latest,
            dof_vel=self.dof_vel,
            dof_tau_est=self.dof_tau_est,
            dof_target=self.filtered_dof_target,
            actions=self.policy.actions,
        )

        if ENABLE_ROS2_LOGGING:
            vx = self.remoteControlService.get_vx_cmd()
            vy = self.remoteControlService.get_vy_cmd()
            vyaw = self.remoteControlService.get_vyaw_cmd()

            self.ctrl_msg.vx = vx
            self.ctrl_msg.vy = vy
            self.ctrl_msg.vyaw = vyaw
            self.pub_control_cmd.publish(self.ctrl_msg)

            self.obs_msg.data = self.policy.obs.tolist()
            self.pub_policy_obs.publish(self.obs_msg)

            self.act_msg.data = self.policy.actions.tolist()
            self.pub_policy_actions.publish(self.act_msg)

        time.sleep(0.001)

    def _publish_cmd(self):
        while self.running:
            time_now = self.timer.get_time()
            if time_now < self.next_publish_time:
                time.sleep(0.001)
                continue
            self.next_publish_time += self.cfg["common"]["dt"]
            self.logger.debug(f"Next publish time: {self.next_publish_time}")

            self.filtered_dof_target = self.filtered_dof_target * 0.8 + self.dof_target * 0.2

            motor_cmd = self.low_cmd.motor_cmd
            for i in range(B1JointCnt):
                motor_cmd[i].q = self.filtered_dof_target[i]

            for i in self.cfg["mech"]["parallel_mech_indexes"]:
                motor_cmd[i].q = self.dof_pos_latest[i]
                motor_cmd[i].tau = np.clip(
                    (self.filtered_dof_target[i] - self.dof_pos_latest[i]) * self.cfg["common"]["stiffness"][i],
                    -self.cfg["common"]["torque_limit"][i],
                    self.cfg["common"]["torque_limit"][i],
                )
                motor_cmd[i].kp = 0.0

            if ENABLE_ROS2_LOGGING:
                self.motor_msg.joint_positions = self.filtered_dof_target.tolist()
                self.motor_msg.joint_velocities = [motor_cmd[i].dq for i in range(B1JointCnt)]
                self.motor_msg.joint_torques = [motor_cmd[i].tau for i in range(B1JointCnt)]
                self.motor_msg.joint_kp = [motor_cmd[i].kp for i in range(B1JointCnt)]
                self.motor_msg.joint_kd = [motor_cmd[i].kd for i in range(B1JointCnt)]
                self.pub_motor_cmd.publish(self.motor_msg)

            start_time = time.perf_counter()
            self._send_cmd(self.low_cmd)
            publish_time = time.perf_counter()
            self.logger.debug(f"Publish took {(publish_time - start_time)*1000:.4f} ms")
            time.sleep(0.001)

    def __enter__(self) -> "Controller":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


if __name__ == "__main__":
    import argparse
    import signal
    import sys

    def signal_handler(sig, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Name of the configuration file.")
    parser.add_argument("--net", type=str, default="127.0.0.1", help="Network interface for SDK communication.")
    parser.add_argument("--profile", type=str, default=None, help="Velocity profile name from command_profiles.yaml (e.g. walk_forward)")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(script_dir, "configs", args.config)

    task_name = os.path.splitext(args.config)[0]

    print(f"Starting custom controller, connecting to {args.net} ...")
    ChannelFactory.Instance().Init(0, args.net)

    with Controller(cfg_file, task_name=task_name) as controller:
        time.sleep(2)
        print("Initialization complete.")
        controller.start_custom_mode_conditionally()
        controller.start_rl_gait_conditionally()

        if args.profile:
            profiles_path = os.path.join(script_dir, "configs", "command_profiles.yaml")
            with open(profiles_path) as f:
                profiles = yaml.safe_load(f)
            if args.profile not in profiles["profiles"]:
                print(f"Error: profile '{args.profile}' not found. Available: {list(profiles['profiles'].keys())}")
                sys.exit(1)
            from utils.command_profile_player import CommandProfilePlayer
            player = CommandProfilePlayer(controller.remoteControlService, profiles["profiles"][args.profile])
            controller.data_logger.file.write(f"# profile: {args.profile}\n")
            controller.data_logger.file.flush()
            player.start()

        try:
            while controller.running:
                controller.run()
            controller.client.ChangeMode(RobotMode.kDamping)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Cleaning up...")
            controller.cleanup()