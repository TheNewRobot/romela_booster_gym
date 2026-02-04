# RoMeLa Booster Gym

Reinforcement learning framework for humanoid robot locomotion on the Booster T1 platform, developed by [RoMeLa](https://www.romela.org/) at UCLA for RoboCup 2026.

Built on [Booster Gym](https://github.com/BoosterRobotics/booster_gym). For our 2024 championship system, see our [RoboCup 2024 paper](https://www.robocup.org/).

[![real_T1_deploy](https://obs-cdn.boosterobotics.com/rl_deploy_demo_video_v3.gif)](https://obs-cdn.boosterobotics.com/rl_deploy_demo_video.mp4)

## Installation

### 1. Create Conda Environment

```bash
conda create --name romela_gym python=3.8
conda activate romela_gym
```

### 2. Install PyTorch with CUDA

**Option A: Conda (recommended)**
```bash
conda install numpy=1.21.6 pytorch=2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Option B: Pip (if conda fails)**
```bash
wget -c https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp38-cp38-linux_x86_64.whl
pip install torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl
pip install torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.21.6
rm torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl
```

### 3. Install Isaac Gym

Download [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym/download), then:

```bash
tar -xzvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e . --no-deps
pip install imageio ninja
cd ../..
```

Fix shared library path:
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
conda deactivate && conda activate romela_gym
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from isaacgym import gymapi; print('Isaac Gym OK')"
```

## Quick Start

### Train
```bash
python scripts/train.py --task=T1 --headless=True
```

Logs saved to `logs/T1/<timestamp>/`. Track progress with TensorBoard:
```bash
tensorboard --logdir logs
```

### Training with Calibrated Physics

To train policies with sysID-calibrated joint dynamics in Isaac Gym:

1. Update `tests/sim2real/config/sim_params.yaml` isaac section with calibrated values
2. Add to `envs/locomotion/T1.yaml`:
```yaml
sim_params:
  use_calibrated: true
  path: "tests/sim2real/config/sim_params.yaml"
```
3. Train: `python scripts/train.py --task=T1`

The calibrated damping, friction, and armature values will be applied to all joints.


### Resume Training
```bash
python scripts/train.py --task=T1 --checkpoint=-1                    # Latest checkpoint
python scripts/train.py --task=T1 --checkpoint=logs/T1/.../model.pth # Specific checkpoint
```

### Play (Isaac Gym)
```bash
python scripts/play.py --task=T1 --num_envs=1 --policy=logs/T1/<exp>/nn/model.pth
```

**Note:** Need a policy to test? Download the [baseline checkpoint](https://drive.google.com/drive/folders/12TgtOTZgYkOfdfY87yxspMiDf8GNqK6m?usp=drive_link) and place it in `logs/T1/`. <exp> corresponds to the name of the experiment.

### Play (MuJoCo)
```bash
python scripts/play_mujoco.py --task=T1 --policy=deploy/models/T1/<exp>.pt
```

### Export & Deploy
```bash
python scripts/export_model.py --checkpoint=logs/T1/<exp>/nn/<exp>.pth
```
See [README_deploy.md](deploy/README_deploy.md) for robot deployment.

## Controls during Play

**Keyboard:**
| Key | Action |
|-----|--------|
| `W/S` | Forward / Backward |
| `A/D` | Strafe Left / Right |
| `Q/E` | Turn Left / Right |
| `Space` | Pause |
| `R` | Reset |
| `O` | Toggle camera follow |

**Xbox Controller:** Left stick = movement, Right stick X = turn.

## Configuration

Training settings: `envs/T1.yaml`

Common overrides:
- `--num_envs=N` — Number of parallel environments
- `--headless=True/False` — Disable/enable visualization
- `--max_iterations=N` — Training iterations
- `--name=exp_name` — Experiment name suffix

## Project Structure

```
romela_booster_gym/
├── envs/                 # Environment configs (T1.yaml, poses)
├── scripts/              # train.py, play.py, export_model.py
├── deploy/               # Robot deployment (see README_deploy.md)
├── tests/
│   ├── policies_arena/   # Policy evaluation (see README_policies_arena.md)
│   └── sim2real/         # Joint dynamics calibration (see README_sim2real.md)
└── resources/            # URDF, meshes, MuJoCo XML
```

## Related Documentation

- [README_deploy.md](deploy/README.md) — Robot deployment instructions
- [README_sim2real.md](tests/sim2real/README.md) — Joint dynamics calibration pipeline
- [README_policies_arena.md](tests/policies_arena/README.md) — Policy evaluation framework

## Related Repositories

- [Booster Gym](https://github.com/BoosterRobotics/booster_gym) — Original training framework
- [Booster Deploy](https://github.com/BoosterRobotics/booster_deploy) — Sim-to-real deployment
- [Booster Assets](https://github.com/BoosterRobotics/booster_assets) — Robot descriptions