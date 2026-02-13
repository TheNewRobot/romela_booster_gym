#!/usr/bin/env python3
"""
Standalone Vicon recorder. Writes base pose to CSV at Vicon frame rate (~200Hz).
Completely independent from deploy_ros.py — runs in a separate terminal.

Creates a timestamped folder under deploy/data/. After the experiment, move
vicon_log.csv into the matching deployment folder.

Requires:
    source /opt/ros/humble/setup.bash
    source ROS2_mocap/ros2/install/setup.bash

Usage:
    cd deploy
    python record_vicon.py
    python record_vicon.py --topic vicon/my_robot/my_robot
"""

import argparse
import csv
import os
import time
import signal
import sys
from datetime import datetime

import rclpy
from rclpy.node import Node
from vicon_receiver.msg import Position


class ViconRecorder(Node):
    def __init__(self, output_dir, topic):
        super().__init__("vicon_recorder")

        os.makedirs(output_dir, exist_ok=True)
        self.filepath = os.path.join(output_dir, "vicon_log.csv")

        self.file = open(self.filepath, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "wall_time", "base_x", "base_y", "base_z",
            "base_qx", "base_qy", "base_qz", "base_qw",
        ])
        self.file.flush()
        self.row_count = 0

        self.subscription = self.create_subscription(
            Position, topic, self.callback, 10
        )

        self.get_logger().info(f"Subscribing to: {topic}")
        self.get_logger().info(f"Logging to: {self.filepath}")

    def callback(self, msg):
        # Position in meters (Vicon sends mm)
        x = msg.x_trans * 0.001
        y = msg.y_trans * 0.001
        z = msg.z_trans * 0.001
        # Quaternion (raw Vicon frame, no transforms)
        qx = msg.x_rot
        qy = msg.y_rot
        qz = msg.z_rot
        qw = msg.w

        self.writer.writerow([
            f"{time.time():.6f}",
            f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
            f"{qx:.6f}", f"{qy:.6f}", f"{qz:.6f}", f"{qw:.6f}",
        ])
        self.row_count += 1
        if self.row_count % 200 == 0:
            self.file.flush()
            self.get_logger().info(f"Recorded {self.row_count} frames")

    def save(self):
        self.file.flush()
        self.file.close()
        self.get_logger().info(f"Saved {self.row_count} rows to {self.filepath}")


def main():
    parser = argparse.ArgumentParser(description="Record Vicon base pose to CSV")
    parser.add_argument("--topic", default="vicon/booster1/booster1", help="Vicon ROS2 topic")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "data", f"vicon_{timestamp}")

    rclpy.init()
    recorder = ViconRecorder(output_dir, args.topic)

    def shutdown(_sig, _frame):
        recorder.save()
        recorder.destroy_node()
        rclpy.shutdown()
        print(f"\nDone. {recorder.row_count} frames saved to: {output_dir}")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    print("Recording... Press Ctrl+C to stop.")
    rclpy.spin(recorder)


if __name__ == "__main__":
    main()
