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

2. **Run deployment:**
   ```bash
   cd deploy
   python deploy.py --config=T1.yaml
   ```

   Options:
   - `--config` — Config file in `configs/` folder
   - `--net` — Network interface (default: `127.0.0.1`, use robot IP for real hardware), we have been using --net=192.168.10.102 when deployed from tegra

3. **Exit safely:**
   - Switch to PREP mode (RT + Y) before terminating the script
   - Never kill the script while robot is in motion

## Configuration

Deployment config: `deploy/configs/T1.yaml`

Key parameters:
- Policy path and observation/action specs
- PD gains (kp, kd)
- Joint limits and safety bounds
- Default pose

### Data Logging

`deploy_ros.py` automatically logs CSV data to `deploy/data/<config_name>/<timestamp>/`.

ROS2 logging is off by default. Set `ENABLE_ROS2_LOGGING = True` in `deploy_ros.py` to re-enable.

**Files generated per run:**

| File | Rate | Contents |
|------|------|----------|
| `commands.csv` | ~50Hz | vx, vy, vyaw |
| `joint_states_actual.csv` | 500Hz | q, dq, tau_est (23 joints) |
| `joint_states_commanded.csv` | 500Hz | q, dq, tau, kp, kd (23 joints) |
| `imu.csv` | ~50Hz | rpy, gyro, acc |
| `policy_obs.csv` | ~50Hz | 47-dim observation |
| `policy_actions.csv` | ~50Hz | 12-dim actions |

### Analyze Deployment

```bash
python tests/sim2real/scripts/plot_deployment.py --data deploy/data/T1/
```

Options:
- `--show` — interactive matplotlib viewer
- `--no-trim` — include standing-still regions

Outputs (saved to same folder):
- `tracking_left_leg.png` / `tracking_right_leg.png` — position tracking + torques per joint
- `overview.png` — commands, vyaw vs gyro_z, IMU
- `stats.txt` — RMS/peak tracking error, torque stats, action smoothness

## Configuration

Deployment config: `deploy/configs/T1.yaml`

Key parameters:
- Policy path and observation/action specs
- PD gains (kp, kd)
- Joint limits and safety bounds
- Default pose
