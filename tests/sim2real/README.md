# Sim2Real Joint Dynamics

Collect joint response data from hanging robot for simulation parameter tuning.

## Network Setup

The robot has multiple computers on internal network (192.168.10.x):
- **Jetson** (where you run scripts)
- **Motion control board** (publishes joint sensor data)

The `--net` flag tells SDK which local interface to bind to:
```bash
--net=192.168.10.102  # Bind to eth0, listen for motion control traffic
```

To find your IP:
```bash
ip addr | grep "192.168"
# Look for eth0 inet address, e.g.: inet 192.168.10.102/24
```

Using `127.0.0.1` (localhost) won't work — it can't see traffic from motion control board.

## Quick Start

```bash
source rcgym.sh
PYTHONPATH=. python tests/sim2real/scripts/sysid_joint_dynamics.py --config=T1.yaml --net=192.168.10.102
```

## Workflow

1. Hang robot securely, put in PREP mode
2. Edit `tests/sim2real/config/sysid_joint_dynamics.yaml` — set joint and experiment name
3. Run script
4. Press **'b'** → custom mode (holds position)
5. Press **'r'** → tests run
6. Data saved automatically
7. **Emergency stop**: PREP button (RT + Y) on robot or damping one (LT + BACK)

## Config

```yaml
experiment:
  name: hanging_test_01  # Creates subfolder

test:
  joint: 11              # Joint to test
  step:
    amplitudes: [0.05, -0.05, 0.1, -0.1]
    duration: 1.0
    settle_time: 1.0
  sine:
    frequencies: [0.5, 1.0, 2.0]
    amplitude: 0.05
    duration: 4.0
```

## Output

```
tests/sim2real/data/
└── hanging_test_01/
    ├── joint_11_Left_Hip_Pitch_20260203_040000.csv
    ├── joint_12_Left_Hip_Roll_20260203_040030.csv
    └── ...
```

CSV format:
```csv
# experiment: hanging_test_01
# joint_index: 11
# joint_name: Left_Hip_Pitch
timestamp,test_type,test_param,cmd_position,actual_position,actual_velocity,actual_torque
```

## Joint Reference

| Index | Joint | Index | Joint |
|-------|-------|-------|-------|
| 11 | Left Hip Pitch | 17 | Right Hip Pitch |
| 12 | Left Hip Roll | 18 | Right Hip Roll |
| 13 | Left Hip Yaw | 19 | Right Hip Yaw |
| 14 | Left Knee Pitch | 20 | Right Knee Pitch |
| 15 | Left Ankle Pitch | 21 | Right Ankle Pitch |
| 16 | Left Ankle Roll | 22 | Right Ankle Roll |