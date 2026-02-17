# Deploy on Booster Robot

Deploy trained policies to the Booster T1 robot via the Booster Robotics SDK.

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install Booster Robotics SDK following the [official guide](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-WDzedC8AiovU8gxSjeGcQ5CInSf).

## Export Model

Convert trained checkpoint to JIT format:

```bash
python scripts/export_model.py --checkpoint=logs/T1/<experiment>/nn/model_XXXX.pth
```

Output: `deploy/models/T1/model_XXXX.pt`

## Test in MuJoCo (Recommended First)

Before deploying to real hardware, validate in MuJoCo:

```bash
python scripts/play_mujoco.py --task=T1 --policy=deploy/models/T1/model_XXXX.pt
```

See [README_main.md](README_main.md#controls) for keyboard and gamepad controls.

### MuJoCo Viewer Tips

| Key | Action |
|-----|--------|
| **Tab** | Toggle GUI panel |
| **F** | Cycle frame visualization |
| **T** | Toggle transparency |
| **C** | Show contact points |

To show world axes: **Tab** → Rendering → Frame → World

## Deploy to Robot

### Safety Controls

| Control | Action |
|---------|--------|
| **RT + Y** | PREP mode (safe idle) |
| **LT + BACK** | Damping mode (soft stop) |
| **E-Stop** | Physical button on back of robot |

**Always know where the e-stop is before running policies.**

### Deployment Steps

1. **Prepare the robot:**
   - **Simulation:** Follow [Webots setup](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-IsE9d2DrIow8tpxCBUUcogdwn5d) or [Isaac setup](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-Jczjd4UKMou7QlxjvJ4c9NNfnwb)
   - **Real robot:** Power on, switch to PREP mode (RT + Y), place on stable ground

2. **Run deployment (basic, no logging):**
   ```bash
   cd deploy
   python deploy.py --config=T1.yaml --net=192.168.10.102
   ```

3. **Run deployment (with CSV logging):**
   ```bash
   cd deploy
   python deploy_ros.py --config=T1.yaml --net=192.168.10.102
   ```

   Options:
   - `--config` — Config file in `configs/` folder
   - `--net` — Network interface (default: `127.0.0.1`, use `192.168.10.102` from tegra)

4. **Safe exit protocol:**
   - Stop commands (press Space)
   - Switch to PREP mode (RT + Y)
   - Kill terminal inmediately (Ctrl+C)
   - Raise the robot 
   - **Never kill the script while robot is in motion**
> **Note** : If the robot finishes in a position that has the legs opened up the motors might overheat when it tries to restore it's PREP position (Heads up!)

## Velocity Profiles

Run reproducible velocity command sequences for benchmarking policies:

```bash
# Profiled run — settle → segments → stop → damp → exit
cd deploy
python deploy_ros.py --config=T1.yaml --net=192.168.10.102 --profile walk_forward

# No profile — keyboard/gamepad as before
python deploy_ros.py --config=T1.yaml --net=192.168.10.102
```

Profiles are defined in `deploy/configs/command_profiles.yaml`. Each profile is a list of `{vx, vy, vyaw, duration}` segments:

```yaml
profiles:
  walk_forward:
    - {vx: 0.0, vy: 0.0, vyaw: 0.0, duration: 2.0}   # settle
    - {vx: 0.3, vy: 0.0, vyaw: 0.0, duration: 5.0}   # walk
    - {vx: 0.0, vy: 0.0, vyaw: 0.0, duration: 2.0}   # stop
```

**Available profiles:** `stand`, `walk_forward`, `walk_backward`, `strafe_left`, `strafe_right`, `turn_left`, `turn_right`, `mixed`

**Execution flow:**
1. Robot initializes and enters RL gait mode
2. 2s settle at zero velocity (automatic)
3. Profile segments play in order
4. After the last segment, the profile thread ends — the main loop keeps running as normal
5. Stop the robot manually as usual (Ctrl+C, gamepad damp, etc.)

**Safety:** All existing safety mechanisms remain active during profiled runs (IMU tilt check, torque limits, gamepad e-stop). The profile only sets velocity commands — it does not modify the control loop.

The deployment CSV automatically logs `# profile: <name>` in the header and records vx/vy/vyaw at every timestep, enabling A/B comparison across policies with identical inputs.

## Configuration

Deployment config: `deploy/configs/T1.yaml`

Key parameters:
- Policy path and observation/action specs
- PD gains (kp, kd)
- Joint limits and safety bounds
- Default pose
- Parallel mechanism ankle indices

## Data Logging

`deploy_ros.py` streams a single CSV at ~50Hz to `deploy/data/<config>/<experiment>/deployment_log.csv`. Data is flushed to disk continuously — survives Ctrl+C or terminal kill with no data loss.

**CSV columns (117 per row):**

| Group | Columns | Count |
|-------|---------|-------|
| Timestamp | `timestamp` | 1 |
| Commands | `vx`, `vy`, `vyaw` | 3 |
| IMU orientation | `roll`, `pitch`, `yaw` | 3 |
| IMU gyro | `gyro_x`, `gyro_y`, `gyro_z` | 3 |
| IMU accelerometer | `acc_x`, `acc_y`, `acc_z` | 3 |
| Joint positions (actual) | `q_act_0` .. `q_act_22` | 23 |
| Joint velocities (actual) | `dq_act_0` .. `dq_act_22` | 23 |
| Joint torques (estimated) | `tau_est_0` .. `tau_est_22` | 23 |
| Joint positions (commanded) | `q_cmd_0` .. `q_cmd_22` | 23 |
| Policy actions | `action_0` .. `action_11` | 12 |

The first line is a comment with the policy name: `# policy: <model_filename>`

**Output directory structure:**
```
deploy/data/T1/
  deploy_log_obs_2026-02-08_04-59-30/
    deployment_log.csv
```

## Analyze Deployment

```bash
cd deploy
python plot_deployment.py --data data/deploy_log_obs_2026-02-08_05-18-58
python plot_deployment.py --data data/deploy_log_obs_2026-02-08_05-18-58 --show
```

Options:
- `--show` — interactive matplotlib viewer
- `--no-trim` — include standing-still regions

Outputs (saved to experiment folder):
- `tracking_left_leg.png` / `tracking_right_leg.png` — position tracking (cmd vs actual) + estimated torques per joint
- `overview.png` — commands timeline, vyaw vs gyro_z, IMU rpy, IMU gyro
- `stats.txt` — per-joint RMS/peak tracking error, torque stats, action smoothness, yaw rate tracking

### Joint Reference

| Leg | Indices | Joint Order |
|-----|---------|-------------|
| Left | 11–16 | Hip_Pitch, Hip_Roll, Hip_Yaw, Knee_Pitch, Ankle_Pitch, Ankle_Roll |
| Right | 17–22 | Hip_Pitch, Hip_Roll, Hip_Yaw, Knee_Pitch, Ankle_Pitch, Ankle_Roll |

Parallel mechanism ankles: joints 15, 16, 21, 22 (torque control, bypass PD).

## Vicon Motion Capture Recording

Record base pose from Vicon alongside deployment data for sim-vs-real walking replay. The Vicon recorder is a **completely separate script** — it does not touch `deploy_ros.py` or any control logic.

### One-time setup

The Vicon bridge uses the `ROS2_mocap` workspace at the repo root.

```bash
# 1. Install Vicon DataStream SDK (see ROS2_mocap/install_DataStreamSDK_10.1/)

# 2. Build the ROS2 workspace
source /opt/ros/humble/setup.bash
cd ROS2_mocap/ros2
colcon build
```

### Recording with Vicon

You need 3 terminals. Connect to WiFi: `romela_apollo_5g` (password: `RoMeLa_Lab_UCLA`).

```bash
# Terminal 1: Vicon bridge (publishes pose to ROS2 topics)
source /opt/ros/humble/setup.bash
cd ROS2_mocap/ros2 && source install/setup.bash
ros2 launch vicon_receiver client.launch.py

# Terminal 2: Vicon CSV recorder (start before deployment, stop after with Ctrl+C)
source /opt/ros/humble/setup.bash
cd ROS2_mocap/ros2 && source install/setup.bash
cd deploy
python record_vicon.py

# Terminal 3: Deployment (unchanged, no ROS2 needed)
conda activate romela_gym
cd deploy
python deploy_ros.py --config=T1.yaml --net=192.168.10.102 --profile walk_forward
```

The Vicon recorder auto-creates a timestamped folder `deploy/data/vicon_<timestamp>/`. After the experiment, move `vicon_log.csv` into the matching deployment folder:
```bash
mv data/vicon_2026-02-15_10-30-00/vicon_log.csv data/deploy_log_obs_2026-02-15_10-30-05/
```

To verify Vicon topics are working (before recording):
```bash
ros2 topic list                           # should show vicon/booster1/booster1
ros2 topic echo vicon/booster1/booster1   # should show live pose data
```

If your Vicon subject has a different name, use `--topic`:
```bash
python record_vicon.py --topic vicon/my_robot/my_robot
```

### Mocap Offset Calibration

The Vicon tracks the L-bracket marker cluster, not the robot's Trunk frame directly. The fixed offset between them is defined in `deploy/configs/mocap_offset.yaml`:

```yaml
dx: -0.07   # meters, backward from trunk
dy:  0.0    # meters, lateral
dz:  0.51   # meters, upward from trunk
pitch: 0.0  # degrees, pitch angle of the bracket
```

To tune this offset visually:

```bash
python deploy/utils/visualize_mocap_frame.py
python deploy/utils/visualize_mocap_frame.py --pitch 5.0
python deploy/utils/visualize_mocap_frame.py --dx -0.07 --dz 0.51 --pitch 3.0
```

This spawns the robot in MuJoCo with an RGB frame (Red=X forward, Green=Y left, Blue=Z up) at the configured offset. Adjust the yaml values until the frame matches where the Vicon markers sit on the real robot. The same config is used by sim2real replay scripts to transform Vicon pose into Trunk pose.

### Testing without the robot

You can verify the Vicon recording works without the T1 — just track any object in Vicon Tracker and run Terminals 1 and 2. Check the output CSV has position data that matches the object's movement.

### Output

After moving `vicon_log.csv` into the deployment folder:
```
deploy/data/deploy_log_obs_2026-02-15_10-30-00/
    deployment_log.csv   # joints, IMU, commands (from deploy_ros.py)
    vicon_log.csv        # base pose at ~200Hz (from record_vicon.py)
```

`vicon_log.csv` columns: `wall_time, base_x, base_y, base_z, base_qx, base_qy, base_qz, base_qw`

Position is in meters (raw Vicon frame, no coordinate transforms). Synchronization with `deployment_log.csv` is done in post-processing by cross-correlating IMU orientation with Vicon orientation.