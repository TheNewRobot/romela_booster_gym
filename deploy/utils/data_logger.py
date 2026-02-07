import os
import csv
import time
import logging


class DataLogger:
    """
    Streaming CSV logger. Writes one row per call to a single file.
    Data is flushed immediately so nothing is lost if the process is killed.
    """

    def __init__(self, output_dir, num_joints=23, num_actions=12):
        self.output_dir = output_dir
        self.num_joints = num_joints
        self.num_actions = num_actions
        self.start_time = time.perf_counter()
        self.logger = logging.getLogger(__name__)
        self.row_count = 0

        os.makedirs(output_dir, exist_ok=True)

        # Build header
        header = ["timestamp", "vx", "vy", "vyaw"]
        header += ["roll", "pitch", "yaw"]
        header += ["gyro_x", "gyro_y", "gyro_z"]
        header += ["acc_x", "acc_y", "acc_z"]
        header += [f"q_act_{i}" for i in range(num_joints)]
        header += [f"dq_act_{i}" for i in range(num_joints)]
        header += [f"tau_est_{i}" for i in range(num_joints)]
        header += [f"q_cmd_{i}" for i in range(num_joints)]
        header += [f"temp_{i}" for i in range(num_joints)]
        header += [f"action_{i}" for i in range(num_actions)]

        filepath = os.path.join(output_dir, "deployment_log.csv")
        self.file = open(filepath, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)
        self.file.flush()
        self.logger.info(f"Logging to {filepath}")

    def _ts(self):
        return time.perf_counter() - self.start_time

    def log(self, vx, vy, vyaw, imu_rpy, imu_gyro, imu_acc,
            dof_pos, dof_vel, dof_tau_est, dof_target, dof_temperature, actions):
        row = [self._ts(), vx, vy, vyaw]
        row += list(imu_rpy)
        row += list(imu_gyro)
        row += list(imu_acc)
        row += list(dof_pos)
        row += list(dof_vel)
        row += list(dof_tau_est)
        row += list(dof_target)
        row += list(dof_temperature)
        row += list(actions)
        self.writer.writerow([f"{v:.6f}" for v in row])
        self.row_count += 1
        if self.row_count % 50 == 0:
            self.file.flush()

    def save(self):
        """Flush and close. Safe to call multiple times."""
        try:
            self.file.flush()
            self.file.close()
            self.logger.info(f"Saved {self.row_count} rows to {self.output_dir}")
        except Exception:
            pass