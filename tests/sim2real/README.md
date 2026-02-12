# Sim2Real Joint Dynamics Calibration

Tune MuJoCo simulation parameters (damping, armature, frictionloss) to match real robot joint dynamics using data from hanging robot tests.

## Overview

The pipeline has three stages:

1. **Data Collection** — Record joint responses on real robot (hanging, no ground contact)
2. **Optimization** — Find simulation parameters that minimize position error vs real data (Nelder-Mead, gradient-free)
3. **Comparison** — Validate tuned simulation against real trajectories, saved as named runs for A/B comparison

## Quick Start

```bash
# 1. Collect data (on robot)
python tests/sim2real/scripts/sysid_joint_dynamics.py --config=T1.yaml --net=192.168.10.102

# 2. Visualize real data (optional, for data quality check)
python tests/sim2real/scripts/plot_joint_response.py --experiment hanging_test_00

# 3. Baseline comparison (before optimization)
python tests/sim2real/scripts/compare_sim_real.py --experiment hanging_test_00 --run-name default_params

# 4. Optimize parameters (all joints, ~10 min)
python tests/sim2real/scripts/optimize_joint_dynamics.py --experiment hanging_test_00

# 5. Compare with optimized params
python tests/sim2real/scripts/compare_sim_real.py --experiment hanging_test_00 --run-name optimized_all
```

### Optimizing a single joint

```bash
python tests/sim2real/scripts/optimize_joint_dynamics.py --experiment hanging_test_00 --joint 15
```

### Using calibrated params in MuJoCo play

```bash
python scripts/play_mujoco.py --task=T1 --policy=deploy/models/T1/<exp>.pt --sim-params
```

Without `--sim-params`, `play_mujoco.py` uses raw XML defaults (zero damping/armature/frictionloss).

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

Nelder-Mead (gradient-free simplex) optimization on MuJoCo parameters to match real joint responses. Physical bounds are enforced to keep parameters in stable ranges.

```bash
# All joints
python tests/sim2real/scripts/optimize_joint_dynamics.py --experiment hanging_test_00

# Single joint
python tests/sim2real/scripts/optimize_joint_dynamics.py --experiment hanging_test_00 --joint 15

# Custom settings
python tests/sim2real/scripts/optimize_joint_dynamics.py --experiment hanging_test_00 --maxfev 400 --subsample 3
```

**Parameters optimized (per-joint):**
| Parameter | Description | Bounds |
|-----------|-------------|--------|
| `damping` | Passive viscous joint damping | [0.1, 10.0] |
| `armature` | Reflected rotor inertia | [0.001, 1.0] |
| `frictionloss` | Dry (Coulomb) friction | [0.001, 2.0] |

**Output:** `tests/sim2real/config/sim_params.yaml` (merged — running `--joint` only updates that joint's entries)

**Typical results (hanging_test_00):**
- Hip joints: damping ~0.5, close to defaults (PD control dominates)
- Yaw/Knee joints: damping 4-6, armature 0.1-0.35 (significant passive dynamics)
- Ankle joints: frictionloss 0.3-2.0 (parallel mechanism adds friction)

## Comparison

Validate tuned parameters by replaying commands in simulation. Each run is saved with a name for A/B comparison.

```bash
# Before optimization
python tests/sim2real/scripts/compare_sim_real.py --experiment hanging_test_00 --run-name default_params

# After optimization
python tests/sim2real/scripts/compare_sim_real.py --experiment hanging_test_00 --run-name optimized_all
```

**Output:**
```
tests/sim2real/data/hanging_test_00/
├── results/
│   ├── default_params.csv       # Metrics before optimization
│   └── optimized_all.csv        # Metrics after optimization
└── plots/comparison/
    ├── default_params/          # Per-joint plots + summary
    └── optimized_all/
```

**Target metrics:**
- RMSE < 0.02 rad (good), < 0.05 rad (acceptable)
- Correlation > 0.95 (good), > 0.8 (acceptable)

## Technical Notes

- **Simulation timing:** The simulation respects real-world timestamps from the data — it runs the correct number of MuJoCo sub-steps between data points to match elapsed real time. This is critical for accurate comparison when subsampling.
- **PD control masking:** Strong PD gains (kp=200 for hips) make some dynamics parameters hard to observe from position data alone. The optimizer finds the best fit within what position tracking can reveal.
- **Ankle limitations:** The parallel mechanism (joints 15-16, 21-22) adds dynamics that 3 scalar parameters can't fully capture. Expect weaker fits for ankle joints.

## File Structure

```
tests/sim2real/
├── config/
│   ├── sysid_joint_dynamics.yaml  # Data collection config
│   ├── sim_params.yaml            # Calibrated parameters (output, git-tracked)
│   ├── comparison.yaml            # Comparison metrics config
│   ├── command_profiles.yaml      # Velocity command profiles
│   └── topics.yaml                # ROS topic mappings
├── data/
│   └── hanging_test_00/           # Experiment data
│       ├── joint_*.csv            # Real robot data
│       ├── results/               # Named comparison metrics
│       └── plots/
│           ├── raw/               # Real data visualization (optional)
│           └── comparison/        # Named sim vs real comparison runs
├── scripts/
│   ├── sysid_joint_dynamics.py    # Data collection (on robot)
│   ├── plot_joint_response.py     # Visualize real data
│   ├── optimize_joint_dynamics.py # Parameter optimization (Nelder-Mead)
│   ├── compare_sim_real.py        # Validation (named runs)
│   └── plot_deployment.py         # Deployment diagnostics
└── utils/
    ├── joint_data_utils.py        # CSV loading, JointData container
    ├── mujoco_utils.py            # MuJoCo simulation (time-aware stepping)
    └── data_utils.py              # General data utilities
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
