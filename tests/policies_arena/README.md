# Policy Arena Evaluation

Automated testing of locomotion policies through progressive difficulty stages across flat, sloped, and rough terrain. Evaluates survival, velocity tracking accuracy, and terrain robustness.

## Quick Start
```bash
# MuJoCo evaluation (primary evaluator)
python tests/policies_arena/scripts/evaluate_mujoco.py --task=T1

# MuJoCo with calibrated joint dynamics
python tests/policies_arena/scripts/evaluate_mujoco.py --task=T1 --sim-params=calibrated

# With visualization (single trial for debugging)
python tests/policies_arena/scripts/evaluate_mujoco.py --task=T1 --headless=false --num-trials=1

# Custom number of trials
python tests/policies_arena/scripts/evaluate_mujoco.py --task=T1 --num-trials=64

# Isaac Gym evaluation (spawns subprocesses per terrain)
python tests/policies_arena/scripts/evaluate.py --task=T1

# Plot results (update TIMESTAMP in script first)
python tests/policies_arena/scripts/plot_results.py
```

## MuJoCo Evaluator (`evaluate_mujoco.py`)

The MuJoCo evaluator is the primary evaluation tool. It runs policies through 10 progressive stages spanning flat, slope, and rough terrain, collecting per-dimension velocity tracking errors and survival rates.

### How It Works

1. **XML Switching**: Automatically switches between `T1_locomotion.xml` (analytical plane, zero contact noise) for flat terrain and `T1_locomotion_arena.xml` (1000x1000 heightfield mesh) for slope and rough terrain. This avoids heightfield collision artifacts on flat stages.

2. **Terrain Generation**: For heightfield stages, terrain data is generated per-trial:
   - **Slope**: 1.5m flat start followed by continuous incline. Gradient defined by `slope` parameter (e.g., 0.087 ≈ 5°, 0.176 ≈ 10°).
   - **Rough**: Random uniform bumps up to `random_height` meters. Each trial uses a different seed for terrain variation.

3. **Velocity Tracking**: Filtered velocity (exponential moving average) is compared against commanded velocity during the metrics window (last portion of each stage). Errors are computed per-dimension (vx, vy, vyaw).

4. **Tracking Modes**: Each terrain defines a `tracking_mode` in the config:
   - `"full"` (default): Pass criteria uses the worst error across all 3 dimensions. Used for flat and slope terrain where the robot should track precisely in all directions.
   - `"commanded"`: Pass criteria only considers dimensions where the command is non-zero. Used for rough terrain where lateral/rotational drift from bump reactions is expected and acceptable.

5. **No Elimination**: All policies run through all stages regardless of pass/fail. Use Ctrl+C to stop early if needed.

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `T1` | Task name (matches config directory) |
| `--config` | `tests/policies_arena/arena_config.yaml` | Path to arena configuration |
| `--headless` | from config | Override: `true` or `false` |
| `--num-trials` | `32` | Trials per policy per stage |
| `--sim-params` | `default` | `default` or `calibrated` (loads from `tests/sim2real/config/sim_params.yaml`) |

### Terminal Output

Each stage prints per-policy results with per-dimension breakdown:
```
[bug_fixed] Stage 7 'Slope Easy': Survival: 100.0%/90%  TrackErr: 0.093/0.130 (full, cmd=0.039) [vx=0.039, vy=0.055, vyaw=0.093]  ✓ PASS
```

- **TrackErr**: Display error / threshold. Display error is `full` or `cmd-only` depending on tracking mode.
- **Parenthetical**: Shows the other tracking metric for reference.
- **Brackets**: Per-dimension mean errors (vx, vy, vyaw).

After all stages, a terrain breakdown table summarizes average tracking error by terrain group:
```
============================================================
          TERRAIN BREAKDOWN (avg tracking error)            
============================================================
  Policy          Flat     Slope     Rough   Overall
  -----------------------------------------------
  baseline       0.078     0.087     0.086     0.081
  sys_ID         0.093     0.127     0.120     0.107
  bug_fixed      0.063     0.097     0.061     0.072
============================================================
```

## Plotting Results

Update the `TIMESTAMP` variable at the top of `plot_results.py` to match your experiment:
```python
TIMESTAMP = "2026-02-04-14-55-33"
```

Then run:
```bash
python tests/policies_arena/scripts/plot_results.py
```

This generates a 2x3 figure (`results_plot.png`) in the experiment directory:
- **Row 1**: Survival rate progression, Vx error heatmap, Vy error heatmap
- **Row 2**: Vyaw error heatmap, Full tracking error heatmap, Terrain breakdown bar chart

## Configuration

Edit `config/arena_config.yaml` to configure:

### Policies
```yaml
policies:
  - name: "my_policy"
    checkpoint: "logs/T1/<experiment>/nn/model_10000.pth"
```

### Terrains
```yaml
terrains:
  flat:
    type: "plane"
    # tracking_mode defaults to "full"
  
  slope_5deg:
    type: "trimesh"
    slope: 0.087            # tan(5°) gradient
    tracking_mode: "full"
  
  rough_2cm:
    type: "trimesh"
    random_height: 0.02     # 2cm random bumps
    tracking_mode: "commanded"
```

### Stages
```yaml
stages:
  - name: "Forward Slow"
    terrain: "flat"
    vx: 0.3
    vy: 0.0
    vyaw: 0.0
    tracking_threshold: 0.110
```

### Timing
```yaml
timing:
  stage_duration_s: 10.0       # Total stage length
  settle_duration_s: 2.0       # Zero-command settling period
  ramp_duration_s: 1.0         # Linear ramp to target velocity
  metrics_window_ratio: 0.5    # Last 50% of stage used for metrics
```

## Pass Criteria

- **Survival**: ≥ threshold (default 90%) of trials must not fall (base height > 0.3m)
- **Tracking** (optional per stage): Tracking error < `tracking_threshold`
  - `tracking_mode: "full"`: max(vx_err, vy_err, vyaw_err) across all dimensions
  - `tracking_mode: "commanded"`: max error only on dimensions with non-zero commands

## Output
```
exps/T1/<timestamp>/
├── results.csv            # Per-stage, per-policy metrics (including vx/vy/vyaw errors)
├── terrain_summary.csv    # Average tracking error per terrain group per policy
├── results_plot.png       # 2x3 visualization figure
├── arena_config.yaml      # Config snapshot
└── task_config.yaml       # Task config snapshot
```

## Evaluators Comparison

| Feature | Isaac Gym (`evaluate.py`) | MuJoCo (`evaluate_mujoco.py`) |
|---------|---------------------------|-------------------------------|
| Parallel envs | Yes (64 default) | No (sequential trials) |
| Subprocess spawning | Yes (one per terrain) | No |
| Calibrated sim params | No | Yes (`--sim-params=calibrated`) |
| Terrain support | Flat, rough, slope | Flat, rough (heightfield), slope (heightfield) |
| XML switching | N/A | Automatic (plane ↔ heightfield) |
| Tracking modes | Full only | Full or commanded (per terrain) |
| Per-dimension errors | No | Yes (vx, vy, vyaw) |
| Terrain breakdown | No | Yes (terminal + CSV) |

## Directory Structure
```
tests/policies_arena/
├── README.md
├── config/
│   └── arena_config.yaml
├── scripts/
│   ├── evaluate.py          # Isaac Gym evaluator
│   ├── evaluate_mujoco.py   # MuJoCo evaluator (primary)
│   └── plot_results.py      # Results visualization
└── utils/
    └── arena_utils.py       # Shared utilities
```