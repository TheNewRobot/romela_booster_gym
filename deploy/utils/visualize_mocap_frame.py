#!/usr/bin/env python3
"""Visualize the Vicon L-bracket mocap frame on the T1 robot in MuJoCo."""

import argparse
import math
import os
import time
import tempfile
import xml.etree.ElementTree as ET

import yaml
import mujoco
import mujoco.viewer


def load_mocap_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "configs", "mocap_offset.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_mocap_config()

    parser = argparse.ArgumentParser(description="Visualize Vicon mocap frame on T1")
    parser.add_argument("--dx", type=float, default=cfg["dx"],
                        help=f"X offset from Trunk in meters (default: {cfg['dx']})")
    parser.add_argument("--dy", type=float, default=cfg["dy"],
                        help=f"Y offset from Trunk in meters (default: {cfg['dy']})")
    parser.add_argument("--dz", type=float, default=cfg["dz"],
                        help=f"Z offset from Trunk in meters (default: {cfg['dz']})")
    parser.add_argument("--pitch", type=float, default=cfg["pitch"],
                        help=f"Pitch angle offset in degrees (default: {cfg['pitch']})")
    args = parser.parse_args()

    # Load the XML
    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "resources", "T1", "T1_locomotion.xml")
    xml_path = os.path.normpath(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Disable gravity so the robot holds its pose
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("gravity", "0 0 0")

    # Find the Trunk body
    trunk = root.find(".//body[@name='Trunk']")

    pitch_rad = math.radians(args.pitch)

    # Add a child body for the mocap frame with RGB axes
    bracket = ET.SubElement(trunk, "body")
    bracket.set("name", "mocap_bracket")
    bracket.set("pos", f"{args.dx} {args.dy} {args.dz}")
    bracket.set("euler", f"0 {pitch_rad} 0")

    # Origin sphere (orange)
    origin = ET.SubElement(bracket, "geom")
    origin.set("type", "sphere")
    origin.set("size", "0.015")
    origin.set("rgba", "1 0.5 0 1")
    origin.set("contype", "0")
    origin.set("conaffinity", "0")

    axis_len = 0.08
    axis_r = "0.004"

    # X axis (red) - forward
    gx = ET.SubElement(bracket, "geom")
    gx.set("type", "cylinder")
    gx.set("size", f"{axis_r} {axis_len / 2}")
    gx.set("pos", f"{axis_len / 2} 0 0")
    gx.set("euler", f"0 {math.pi / 2} 0")
    gx.set("rgba", "1 0 0 0.9")
    gx.set("contype", "0")
    gx.set("conaffinity", "0")

    # Y axis (green) - left
    gy = ET.SubElement(bracket, "geom")
    gy.set("type", "cylinder")
    gy.set("size", f"{axis_r} {axis_len / 2}")
    gy.set("pos", f"0 {axis_len / 2} 0")
    gy.set("euler", f"{math.pi / 2} 0 0")
    gy.set("rgba", "0 1 0 0.9")
    gy.set("contype", "0")
    gy.set("conaffinity", "0")

    # Z axis (blue) - up
    gz = ET.SubElement(bracket, "geom")
    gz.set("type", "cylinder")
    gz.set("size", f"{axis_r} {axis_len / 2}")
    gz.set("pos", f"0 0 {axis_len / 2}")
    gz.set("rgba", "0 0 1 0.9")
    gz.set("contype", "0")
    gz.set("conaffinity", "0")

    # Write modified XML to temp file in same directory (so meshdir resolves)
    xml_dir = os.path.dirname(xml_path)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=xml_dir)
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(ET.tostring(root, encoding="unicode"))

        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Set standing pose: free joint qpos = [x, y, z, qw, qx, qy, qz]
    data.qpos[2] = 0.7
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)

    print(f"Config: deploy/configs/mocap_offset.yaml")
    print(f"Mocap frame offset from Trunk: dx={args.dx}, dy={args.dy}, dz={args.dz}, pitch={args.pitch}deg")
    print(f"RGB axes: Red=X (forward), Green=Y (left), Blue=Z (up)")
    print(f"Orange sphere = mocap frame origin")

    # Right-side camera view looking at the head
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = -90     # right side
        viewer.cam.elevation = -5
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = [0, 0, 0.9]  # head height
        viewer.sync()
        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()
