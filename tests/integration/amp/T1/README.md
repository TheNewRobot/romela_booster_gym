# AMP - Booster T1

Reference motion data and replay tools for AMP (Adversarial Motion Priors) on the Booster T1.

## Config

All settings (dataset files, joint name mapping, camera) live in `config/config.yaml`.

## Download motions

```bash
python tests/integration/amp/T1/scripts/download_dataset.py
```

Downloads `.npz` clips from HuggingFace into `data/raw/`. File list is defined in `config/config.yaml`.

## Replay in MuJoCo

```bash
# Just provide the filename
python tests/integration/amp/T1/scripts/replay_motion_mujoco.py --file walking1.npz

# Specific trajectory segment
python tests/integration/amp/T1/scripts/replay_motion_mujoco.py --file goal_kick.npz --traj 0

# Slow motion
python tests/integration/amp/T1/scripts/replay_motion_mujoco.py --file walking1.npz --speed 0.5

# Headless data collection to CSV
python tests/integration/amp/T1/scripts/replay_motion_mujoco.py --file walking1.npz --collect --headless
```

Viewer controls: `[Space]` pause/resume, `[R]` restart, `[O]` toggle camera follow/free, `[Esc]` quit.

Contact forces: press `[F]` or right-click → Rendering → **Contact Force** / **Contact Point**.

## Inspect a motion file

```bash
python tests/integration/amp/T1/scripts/inspect_npz.py --file walking1.npz
```

Prints shapes, joint names, frequencies, split points, and duration.
