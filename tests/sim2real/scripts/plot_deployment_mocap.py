#!/usr/bin/env python3
"""Plot Vicon base trajectory from a deployment experiment."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.deployment_utils import (
    load_deployment_data, load_vicon_data, find_vicon_data,
    align_timestamps, resample_vicon_to_deployment,
    transform_positions_to_body_frame,
)
from utils.deployment_plots import plot_deployment_mocap
from utils.data_utils import load_yaml


def main():
    parser = argparse.ArgumentParser(description='Plot deployment + Vicon data')
    parser.add_argument('--deploy', required=True, help='Path to deployment folder')
    parser.add_argument('--task-config', default='envs/locomotion/T1.yaml')
    args = parser.parse_args()

    deploy_dir = Path(args.deploy)
    deploy_csv = deploy_dir / 'deployment_log.csv'
    output_dir = deploy_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(Path(args.task_config))
    deploy = load_deployment_data(str(deploy_csv), cfg)
    print(f"Loaded {len(deploy.timestamps)} frames, {deploy.timestamps[-1]:.1f}s")

    vicon_path = find_vicon_data(str(deploy_dir))
    if not vicon_path:
        print(f"No Vicon data found in {deploy_dir}")
        return

    vicon = load_vicon_data(vicon_path)
    offset, vicon_aligned = align_timestamps(deploy, vicon)
    pos, quat = resample_vicon_to_deployment(deploy, vicon, vicon_aligned)
    body = transform_positions_to_body_frame(pos, quat[0], quat_format="xyzw")

    plot_deployment_mocap(
        deploy.timestamps, body,
        deploy.vx, deploy.vy, deploy.vyaw,
        output_dir,
    )
    print(f"Saved xy_path.png, timeseries.png (vicon offset: {offset:.2f}s)")


if __name__ == '__main__':
    main()
