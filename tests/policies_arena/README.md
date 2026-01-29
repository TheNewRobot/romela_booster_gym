
Readme · MD
Copy

# Policy Arena Evaluation

Automated testing of locomotion policies through progressive difficulty stages. Policies that fail a stage are eliminated; survivors advance.

## Quick Start

```bash
# Run evaluation with default config
python tests/policies_arena/evaluate.py --task=T1

# Run with visualization (first policy only, 1 env)
python tests/policies_arena/evaluate.py --task=T1 --headless=false

# Use custom config
python tests/policies_arena/evaluate.py --task=T1 --config=path/to/custom_config.yaml
```

## Configuration

Edit `arena_config.yaml` to configure:

- **policies**: List of checkpoints to compare
- **evaluation**: Number of envs, survival threshold, seed
- **timing**: Stage duration, settle time, ramp time
- **terrains**: Define terrain types (flat, slope, rough)
- **stages**: Define test stages (terrain + velocity commands + pass threshold)

## Output

Each run creates:
```
exp/T1/<timestamp>/
├── results.csv      # Metrics per (policy, stage)
└── config.yaml      # Config snapshot for reproducibility
```

### CSV Columns
| Column | Description |
|--------|-------------|
| policy_name | Policy identifier |
| stage | Stage number (1-indexed) |
| stage_name | Stage display name |
| terrain | Terrain type used |
| vx_cmd, vy_cmd, vyaw_cmd | Commanded velocities |
| survival_rate | Fraction of envs that survived |
| vx_error, vy_error, vyaw_error | Mean tracking errors |
| mean_height | Average base height |
| mean_power | Average power consumption |
| passed | true/false |

## Pass Criteria

- **Survival**: ≥90% of environments must not trigger reset
- **Tracking** (if threshold set): Max velocity error < threshold

## Notes

- Isaac Gym only allows one simulation per process, so the script spawns subprocesses for each terrain type automatically
- Stages using the same terrain run in a single subprocess (faster)
- Metrics are computed over the last 50% of each stage (steady-state)