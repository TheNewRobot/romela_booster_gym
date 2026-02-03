# Sim2Real Joint Dynamics Calibration

Tune MuJoCo simulation parameters (damping, armature, frictionloss) to match real robot joint dynamics using data from hanging robot tests.

## Overview

The pipeline has three stages:

1. **Data Collection** — Record joint responses on real robot (hanging, no ground contact)
2. **Optimization** — Find simulation parameters that minimize error vs real data
3. **Comparison** — Validate tuned simulation against real trajectories

## Quick Start

```bash
# 1. Collect data (on robot)
python tests/sim2real/scripts/sysid_joint_dynamics.py --config=T1.yaml --net=192.168.10.102

# 2. Visualize real data
python tests/sim2real/scripts/plot_joint_response.py --experiment hanging_test_00

# 3. Optimize parameters
python tests/sim2real/scripts/optimize_joint_dynamics.py --experiment hanging_test_00 --iterations 1000 --lr 0.1

# 4. Compare sim vs real
python tests/sim2real/scripts/compare_sim_real.py --experiment hanging_test_00
```

## Data Collection

### Network Setup

The robot has multiple computers on internal network (192.168.10.x). The `--net` flag specifies which interface to bind:

```bash
ip addr | grep "192.168"   # Find your IP (e.g., 192.168.10.102)
--net=192.168.10.102       # Use eth0, NOT localhost, this is the default address
```

### Test Protocol

For each joint (robot hanging, no ground contact):
- **Step responses:** ±0.05, ±0.1 rad from neutral (2s hold each)
- **Sinusoid sweeps:** 0.5, 1.0, 2.0 Hz at ±0.1 rad amplitude (4s each)

### Running Tests

1. Hang robot securely, put in PREP mode
2. Edit `tests/sim2real/config/sysid_joint_dynamics.yaml`:
   ```yaml
   experiment:
     name: hanging_test_00
   test:
     joint: 11  # Joint to test (11-22 for legs)
   ```
3. Run: `python tests/sim2real/scripts/sysid_joint_dynamics.py --config=T1.yaml --net=<IP>`
4. Press **'b'** → custom mode, then **'r'** → run tests
5. **Emergency stop:** PREP button (RT + Y) or damping (LT + BACK)

### Ankle Joints (Parallel Mechanism)

Joints 15-16 and 21-22 use parallel mechanisms. The script automatically activates both ankle motors when testing either ankle joint. No special configuration needed.

### Output

```
tests/sim2real/data/hanging_test_00/
├── joint_11_Left_Hip_Pitch.csv
├── joint_12_Left_Hip_Roll.csv
└── ...
```

CSV columns: `timestamp, test_type, test_param, cmd_position, actual_position, actual_velocity, actual_torque`

## Optimization

Gradient descent on MuJoCo parameters to match real joint responses.

```bash
python tests/sim2real/scripts/optimize_joint_dynamics.py \
    --experiment hanging_test_00 \
    --iterations 500 \
    --lr 0.005
```

**Parameters optimized (per-joint):**
- `damping` — Passive joint damping
- `armature` — Reflected rotor inertia  
- `frictionloss` — Dry friction

**Output:** `tests/sim2real/config/sim_params.yaml`

## Comparison

Validate tuned parameters by replaying commands in simulation:

```bash
python tests/sim2real/scripts/compare_sim_real.py --experiment hanging_test_00
```

**Output:**
- Per-joint comparison plots: `plots/comparison/comparison_joint_*.png`
- Summary plot: `plots/comparison/comparison_summary.png`
- Metrics CSV: `comparison_metrics.csv`

**Target metrics:**
- RMSE < 0.05 rad
- Correlation > 0.8

## File Structure

```
tests/sim2real/
├── config/
│   ├── sysid_joint_dynamics.yaml  # Data collection config
│   └── sim_params.yaml            # Optimized parameters (output)
├── data/
│   └── hanging_test_00/           # Example experiment
│       ├── joint_*.csv            # Real robot data
│       ├── comparison_metrics.csv
│       └── plots/
│           ├── raw/               # Real data visualization
│           └── comparison/        # Sim vs real plots
├── scripts/
│   ├── sysid_joint_dynamics.py    # Data collection (on robot)
│   ├── plot_joint_response.py     # Visualize real data
│   ├── optimize_joint_dynamics.py # Parameter optimization
│   └── compare_sim_real.py        # Validation
└── utils/
    ├── joint_data_utils.py        # CSV loading
    └── mujoco_utils.py            # MuJoCo simulation helpers
```

## Joint Reference

| Index | Left Leg | Index | Right Leg |
|-------|----------|-------|-----------|
| 11 | Hip Pitch | 17 | Hip Pitch |
| 12 | Hip Roll | 18 | Hip Roll |
| 13 | Hip Yaw | 19 | Hip Yaw |
| 14 | Knee Pitch | 20 | Knee Pitch |
| 15 | Ankle Pitch* | 21 | Ankle Pitch* |
| 16 | Ankle Roll* | 22 | Ankle Roll* |

*Parallel mechanism — both motors activated during testing as they are coupled
