# RoMeLa Booster Gym

A reinforcement learning framework for humanoid robot locomotion on the Booster T1 platform, developed by the [RoMeLa](https://www.romela.org/) team at UCLA for RoboCup 2026.

This repository builds upon [Booster Gym](https://github.com/BoosterRobotics/booster_gym) and integrates with our RoboCup software stack. For details on our 2024 championship system, see:

> **A Hierarchical, Model-Based System for High-Performance Humanoid Soccer**  
> Wang Q.*, Zhu M.*, Hou R.*, Gillespie K., Zhu A., Wang S., Wang Y., Fernandez G.I., Liu Y., Togashi C., Nam H., Navghare A., Xu A., Zhu T., Ahn M.S., **Flores Alvarez A.**, Quan J., Hong E., Hong D.W.  
> *RoboCup 2024 Adult-Sized Humanoid Soccer Champions*

**Related Booster Robotics Resources:**
- [Booster Train](https://github.com/BoosterRobotics/booster_train) - RL tasks using Isaac Lab
- [Booster Deploy](https://github.com/BoosterRobotics/booster_deploy) - Sim-to-real deployment framework
- [Booster Assets](https://github.com/BoosterRobotics/booster_assets) - Robot descriptions and motion data

[![real_T1_deploy](https://obs-cdn.boosterobotics.com/rl_deploy_demo_video_v3.gif)](https://obs-cdn.boosterobotics.com/rl_deploy_demo_video.mp4)

## Features

- **Complete Training-to-Deployment Pipeline**: Full support for training, evaluating, and deploying policies in simulation and on real robots.
- **Sim-to-Real Transfer**: Including effective settings and techniques to minimize the sim-to-real gap and improve policy generalization.
- **Customizable Environments and Algorithms**: Easily modify environments and RL algorithms to suit a wide range of tasks.
- **Out-of-the-Box Booster T1 Support**: Pre-configured for quick setup and deployment on the Booster T1 robot.

## Overview

The framework supports the following stages for reinforcement learning:

1. **Training**: 

    - Train reinforcement learning policies using Isaac Gym with parallelized environments.

2. **Playing**:

    - **In-Simulation Testing**: Evaluate the trained policy in the same environment with training to ensure it behaves as expected.
    - **Cross-Simulation Testing**: Test the policy in MuJoCo to verify its generalization across different environments.

3. **Deployment**:

    - **Model Export**: Export the trained policy from `*.pth` to a JIT-optimized `*.pt` format for efficiency deployment
    - **Webots Deployment**: Use the SDK to deploy the model in Webots for final verification in simulation.
    - **Physical Robot Deployment**: Deploy the model to the physical robot using the same Webots deployment script.

## Installation

Follow these steps to set up your environment:

### 1. Create Conda Environment

```sh
conda create --name romela_gym python=3.8
conda activate romela_gym
```

### 2. Install PyTorch with CUDA Support

**Option A: Conda (recommended if connection is stable)**
```sh
conda install numpy=1.21.6 pytorch=2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Option B: Wget + Pip (for unreliable connections)**

If conda downloads fail due to connection issues, use wget with resume capability:
```sh
# Download PyTorch wheel (supports resume with -c flag)
wget -c https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp38-cp38-linux_x86_64.whl

# Install from local file
pip install torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl
pip install torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.21.6

# Clean up downloaded files
rm torch-2.0.1+cu118-cp38-cp38-linux_x86_64.whl
```

### 3. Install Isaac Gym

Download Isaac Gym Preview 4 from [NVIDIA's website](https://developer.nvidia.com/isaac-gym/download).

Extract and install:
```sh
tar -xzvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e . --no-deps  # --no-deps prevents overwriting PyTorch
pip install imageio ninja   # Install missing dependencies
cd ../..
```

Configure the environment to handle shared libraries (fixes `libpython3.8` not found):
```sh
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo 'unset OLD_LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# Reactivate to apply changes
conda deactivate && conda activate romela_gym
```

Alternatively, edit the files manually with `nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`.

### 4. Install Python Dependencies

```sh
pip install -r requirements.txt
```

### 5. Install Package (Editable Mode)
```sh
pip install -e .
```

This adds the project root to Python's path permanently.

### 6. Verify Installation

```sh
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "from isaacgym import gymapi; print('Isaac Gym OK')"
```

## Usage

### 1. Training

To start training a policy, run the following command:

```sh
$ python scripts/train.py --task=T1 --headless=True
```

Training logs and saved models will be stored in `logs/<date-time>/`.

#### Configurations

Training settings are loaded from `envs/<task>.yaml`. You can also override config values using command-line arguments:

- `--checkpoint`: Path of the model checkpoint to load (set to `-1` to use the most recent model).
- `--num_envs`: Number of environments to create.
- `--headless`: Run headless without creating a viewer window.
- `--sim_device`: Device for physics simulation (e.g., `cuda:0`, `cpu`). 
- `--rl_device`: Device for the RL algorithm (e.g., `cuda:0`, `cpu`). 
- `--seed`: Random seed.
- `--max_iterations`: Maximum number of training iterations.
- `--name`: Experiment name suffix for the log folder.

#### Resuming Training from Checkpoint

To continue training from a previous run, use the `--checkpoint` argument:
```sh
# Resume from the most recent checkpoint
$ python scripts/train.py --task=T1 --checkpoint=-1

# Resume from a specific checkpoint
$ python scripts/train.py --task=T1 --checkpoint=logs/T1/2026-01-27-00-28-31/nn/model_500.pth
```

**What gets loaded:**
- Model weights
- Optimizer state (learning rate, momentum, etc.)
- Curriculum progress (if using curriculum learning)

**Note:** Resuming creates a new log folder. The iteration counter resets to 0, but training continues from the loaded weights. Use `--name` to label resumed experiments:
```sh
$ python scripts/train.py --task=T1 --checkpoint=-1 --name=resumed_higher_vel
```

To add a new task, create a config file in `envs/` and register the environment in `envs/__init__.py`.

#### Progress Tracking

To visualize training progress with [TensorBoard](https://www.tensorflow.org/tensorboard), run:

```sh
$ tensorboard --logdir logs
```

To use [Weights & Biases](https://wandb.ai/) for tracking, log in first:

```sh
$ wandb login
```

You can disable W&B by setting `use_wandb` to `false` in the config file.

---

### 2. Playing

#### In-Simulation Testing

To test the trained policy in Isaac Gym with a single robot (recommended for visualization):
```sh
python scripts/play.py --task=T1 --checkpoint=-1 --num_envs=1 --terrain=plane
```

> **Note:** Use `--num_envs=1` when visualizing. Use `--terrain=plane` for flat ground (default is `trimesh` from config).

Videos of the evaluation are automatically saved in `videos/<date-time>.mp4`. You can disable video recording by setting `record_video` to `false` in the config file.

#### Cross-Simulation Testing

To test the policy in MuJoCo (sim-to-sim transfer validation):

```sh
python scripts/play_mujoco.py --task=T1 --checkpoint=-1
```

---
> **Note:** If you want a policy to test under logs/T1, you can try this downloading this one from GD called baseline ([link](https://drive.google.com/drive/folders/12TgtOTZgYkOfdfY87yxspMiDf8GNqK6m?usp=drive_link))

### 3. Deployment

To deploy a trained policy through the Booster Robotics SDK in simulation or in the real world, export the model using:

```sh
$ python scripts/export_model.py --task=T1 --checkpoint=-1
```

After exporting the model, follow the steps in [Deploy on Booster Robot](deploy/README.md) to complete the deployment process.

### 4. Visualize Poses

Check robot poses and spawn heights without running a full policy:
```sh
python scripts/visualize_poses.py --task=T1 --list              # List available poses
python scripts/visualize_poses.py --task=T1                     # Default pose
python scripts/visualize_poses.py --task=T1 --pose=crouch --height=0.55
```

Poses are defined in `envs/locomotion/t1_poses.yaml`. Useful for checking floor collisions at different heights.

---

## RoboCup 2025 Roadmap

This repository is being extended to support multiple RL policies for RoboCup 2025 humanoid soccer. The goal is to replace the current model-based locomotion with learned policies while integrating with the existing behavior tree stack.

### Planned Policies

| Policy | Description | Status |
|--------|-------------|--------|
| **Locomotion (LP)** | Velocity-command walking/running | In Progress |
| **Dribbling (DP)** | Ball control while moving toward goal | Planned |
| **Kicking (KP)** | Powerful kicks from various positions | Planned |
| **Goalkeeper (GP)** | Defensive positioning and saves | Planned |

### Integration Architecture

```
Behavior Tree (existing)
    │
    ├── Ball Memory Manager
    ├── Role Assignment  
    ├── Kick Target Selection
    │
    ▼
RL Policy Interface (velocity commands: vx, vy, vyaw)
    │
    ├── Locomotion Policy (LP)
    ├── Dribbling Policy (DP)  
    ├── Kicking Policy (KP)
    └── Goalkeeper Policy (GP)
    │
    ▼
Low-Level Control → Booster T1 Hardware
```

### Development Priorities

1. **Sim2Sim Validation** - Transfer policies from Isaac Gym to MuJoCo for testing
2. **Policy Interface Definition** - Standardize observation/action spaces for BT integration
3. **Benchmark Suite** - Create test scenarios to measure policy improvements
4. **Deploy/Fine-tune** - Real robot deployment and domain adaptation

### Repository Structure (Planned)

```
romela_booster_gym/
├── resources/        # CAD Files, expert data
├── envs/             # Environment definitions (LP, DP, KP, GP)
├── policies/         # Neural network architectures
├── runners/          # Training and evaluation scripts
├── deploy/           # Real robot deployment
├── tests/            # Unit, integration, sim2sim tests
└── external/         # Imported code (dribblebot, goalkeeper, etc.)