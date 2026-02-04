# Policy Arena Evaluation

Automated testing of locomotion policies through progressive difficulty stages. Policies that fail a stage are eliminated; survivors advance.

## Quick Start
```bash
# Isaac Gym evaluation (spawns subprocesses per terrain)
python tests/policies_arena/scripts/evaluate.py --task=T1

# MuJoCo evaluation (single process, supports calibrated sim params)
python tests/policies_arena/scripts/evaluate_mujoco.py --task=T1

# MuJoCo with calibrated joint dynamics
python tests/policies_arena/scripts/evaluate_mujoco.py --task=T1 --sim-params=calibrated

# With visualization
python tests/policies_arena/scripts/evaluate_mujoco.py --task=T1 --headless=false --num-trials=1
```

## Evaluators

| Feature | Isaac Gym (`evaluate.py`) | MuJoCo (`evaluate_mujoco.py`) |
|---------|---------------------------|-------------------------------|
| Parallel envs | Yes (64 default) | No (sequential trials) |
| Subprocess spawning | Yes (one per terrain) | No |
| Calibrated sim params | No | Yes (`--sim-params=calibrated`) |
| Rough terrain | Yes | Limited (see below) |

## Configuration

Edit `config/arena_config.yaml` to configure:
- **policies**: Checkpoints to compare
- **evaluation**: Survival threshold, seed, headless
- **timing**: Stage duration, settle/ramp times
- **terrains**: Terrain type definitions
- **stages**: Test stages (terrain + velocity commands)

## Output
```
exps/T1/<timestamp>/
├── results.csv          # Metrics per (policy, stage)
├── arena_config.yaml    # Config snapshot
└── task_config.yaml     # Task config snapshot
```

## Pass Criteria

- **Survival**: ≥90% of trials must not fall, this can be tuned
- **Tracking** (optional): Max velocity error < threshold


## Directory Structure
```
tests/policies_arena/
├── README.md
├── config/
│   └── arena_config.yaml
├── scripts/
│   ├── evaluate.py          # Isaac Gym evaluator
│   ├── evaluate_mujoco.py   # MuJoCo evaluator
│   └── plot_results.py      # Results visualization
└── utils/
    └── arena_utils.py       # Shared utilities
```