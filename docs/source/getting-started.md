# Getting Started

## Prerequisites

AGILE runs on top of [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html). All Isaac Lab requirements (Isaac Sim 5.1, NVIDIA GPU, etc.) apply. See the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html).

```{warning}
AGILE requires Isaac Lab **v2.3.2** specifically. Other versions may have incompatible APIs.
```

## Installation

### 1. Install Isaac Lab

Follow the [Isaac Lab binaries installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/binaries_installation.html) using the pre-built Isaac Sim binaries. In summary:

```bash
# 1. Download and extract Isaac Sim pre-built binaries
#    https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html

# 2. Clone Isaac Lab and check out the required version
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab
git checkout v2.3.2

# 3. Create symlink to Isaac Sim
ln -s /path/to/isaac-sim _isaac_sim

# 4. Create the conda environment
./isaaclab.sh --conda agile_env
conda activate agile_env

# 5. Install dependencies (Linux)
sudo apt install cmake build-essential

# 6. Install Isaac Lab extensions
./isaaclab.sh --install
```

### 2. Install AGILE

With the `agile_env` conda environment active:

```bash
conda activate agile_env
export ISAACLAB_PATH=/path/to/IsaacLab

git clone git@github.com:nvidia-isaac/WBC-AGILE.git agile && cd agile
git lfs pull

# Install AGILE + dependencies into the Isaac Lab environment
./scripts/setup/install_deps_local.sh

# Verify custom rsl_rl is installed correctly
python scripts/verify_rsl_rl.py
```

### 3. Set up pre-commit hooks (optional, for development)

```bash
./scripts/setup/setup_hooks.sh
pre-commit run --all-files  # verify
```

## Quick Start

```bash
# Validate an environment (no trained policy needed)
python scripts/play.py --task Velocity-T1-v0 --num_envs 2

# Train a velocity tracking policy
python scripts/train.py \
    --task Velocity-T1-v0 \
    --num_envs 2048 \
    --headless \
    --logger wandb

# Evaluate a trained policy
python scripts/eval.py \
    --task Velocity-T1-v0 \
    --checkpoint /path/to/model.pt \
    --num_envs 32
```

See {doc}`training` for the full training guide, and {doc}`pretrained-policies` for available checkpoints and policy formats.

## Project Structure

```
agile/
├── agile/
│   ├── algorithms/
│   │   ├── rsl_rl/          # Custom RSL-RL with TensorDict support
│   │   └── evaluation/      # Evaluation metrics and reporting
│   ├── data/policy/          # Pre-trained checkpoints
│   ├── isaaclab_extras/      # Isaac Lab extensions
│   ├── sim2mujoco/           # Sim-to-MuJoCo transfer
│   └── rl_env/
│       ├── assets/           # Robot USD assets
│       ├── mdp/              # MDP components (rewards, actions, observations, ...)
│       ├── tasks/            # Task definitions (self-contained configs)
│       └── rsl_rl/           # RSL-RL integration wrappers
├── scripts/                  # train.py, eval.py, play.py, export, setup
├── tests/                    # Unit and E2E tests
├── workflows/                # Docker + OSMO remote training configs
└── run.py                    # CLI for remote OSMO training/eval
```
