import os
import csv
import time
import threading
import logging


class DataLogger:
    def __init__(self, output_dir, num_joints=23):
        self.output_dir = output_dir
        self.num_joints = num_joints
        self.start_time = time.perf_counter()
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

        os.makedirs(output_dir, exist_ok=True)

        self.commands = []
        self.joint_actual = []
        self.joint_commanded = []
        self.imu = []
        self.policy_obs = []
        self.policy_actions = []

    def _ts(self):
        return time.perf_counter() - self.start_time

    def log_command(self, vx, vy, vyaw):
        with self.lock:
            self.commands.append([self._ts(), vx, vy, vyaw])

    def log_joint_actual(self, q, dq, tau_est):
        with self.lock:
            self.joint_actual.append([self._ts()] + list(q) + list(dq) + list(tau_est))

    def log_joint_commanded(self, q, dq, tau, kp, kd):
        with self.lock:
            self.joint_commanded.append(
                [self._ts()] + list(q) + list(dq) + list(tau) + list(kp) + list(kd)
            )

    def log_imu(self, rpy, gyro, acc):
        with self.lock:
            self.imu.append([self._ts()] + list(rpy) + list(gyro) + list(acc))

    def log_policy(self, obs, actions):
        with self.lock:
            self.policy_obs.append([self._ts()] + list(obs))
            self.policy_actions.append([self._ts()] + list(actions))

    def save(self):
        n = self.num_joints

        self._write_csv("commands.csv", ["timestamp", "vx", "vy", "vyaw"], self.commands)

        header = ["timestamp"]
        header += [f"q_{i}" for i in range(n)]
        header += [f"dq_{i}" for i in range(n)]
        header += [f"tau_{i}" for i in range(n)]
        self._write_csv("joint_states_actual.csv", header, self.joint_actual)

        header = ["timestamp"]
        header += [f"q_{i}" for i in range(n)]
        header += [f"dq_{i}" for i in range(n)]
        header += [f"tau_{i}" for i in range(n)]
        header += [f"kp_{i}" for i in range(n)]
        header += [f"kd_{i}" for i in range(n)]
        self._write_csv("joint_states_commanded.csv", header, self.joint_commanded)

        self._write_csv(
            "imu.csv",
            ["timestamp", "roll", "pitch", "yaw", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"],
            self.imu,
        )

        if self.policy_obs:
            n_obs = len(self.policy_obs[0]) - 1
            self._write_csv(
                "policy_obs.csv",
                ["timestamp"] + [f"obs_{i}" for i in range(n_obs)],
                self.policy_obs,
            )

        if self.policy_actions:
            n_act = len(self.policy_actions[0]) - 1
            self._write_csv(
                "policy_actions.csv",
                ["timestamp"] + [f"action_{i}" for i in range(n_act)],
                self.policy_actions,
            )

        self.logger.info(f"Saved deployment data to {self.output_dir}")
        self.logger.info(
            f"  commands: {len(self.commands)}, joint_actual: {len(self.joint_actual)}, "
            f"joint_commanded: {len(self.joint_commanded)}, imu: {len(self.imu)}, "
            f"policy_obs: {len(self.policy_obs)}, policy_actions: {len(self.policy_actions)}"
        )

    def _write_csv(self, filename, header, rows):
        if not rows:
            return
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow([f"{v:.6f}" for v in row])