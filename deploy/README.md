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

ROS2 logging is off by default. Set `ENABLE_ROS2_LOGGING = True` in `deploy_ros.py` to re-enable (requires ROS2 workspace sourced).

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
python tests/sim2real/scripts/plot_deployment.py --data deploy/data/T1/deploy_log_obs_2026-02-08_04-59-30
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