# Getting Started

## Prerequisites

AGILE runs on top of [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html). All Isaac Lab requirements (Isaac Sim 6.0, Python 3.12, NVIDIA GPU, etc.) apply. See the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). [uv](https://docs.astral.sh/uv/) is used for environment and dependency management throughout.

```{warning}
AGILE requires Isaac Lab **v3.0.0-beta2** (Isaac Sim 6.0, Python 3.12). Other versions (e.g. v2.3.2 / Isaac Sim 5.1) have incompatible APIs.
```

## Installation

### 1. Install AGILE

```bash
git clone <repository-url> agile && cd agile
git lfs pull

# Install AGILE's pinned uv environment.
uv sync --frozen

# Verify public rsl-rl-lib 5.4.1 plus the AGILE patch
uv run scripts/verify_rsl_rl.py
```

The locked environment installs Isaac Lab `v3.0.0-beta2`, Isaac Sim 6.0, LEAPP, RSL-RL, and AGILE's Python dependencies.

### 2. Set up pre-commit hooks (optional, for development)

```bash
./scripts/setup/setup_hooks.sh
pre-commit run --all-files  # verify
```

## Quick Start

```bash
# Validate an environment (no trained policy needed)
uv run scripts/play.py --task Velocity-T1-v0 --num_envs 2

# Train a velocity tracking policy
uv run scripts/train.py \
    --task Velocity-T1-v0 \
    --num_envs 2048 \
    --headless \
    --logger wandb

# Evaluate a trained policy
uv run scripts/eval.py \
    --task Velocity-T1-v0 \
    --checkpoint /path/to/model.pt \
    --num_envs 32
```

See {doc}`training` for the full training guide, and {doc}`pretrained-policies` for available checkpoints and policy formats.

## Project Structure

```
agile/
+-- agile/
|   +-- algorithms/
|   |   +-- evaluation/      # Evaluation metrics and reporting
|   +-- data/policy/         # Pre-trained checkpoints
|   +-- isaaclab_extras/     # Isaac Lab extensions
|   +-- sim2mujoco/          # Sim-to-MuJoCo transfer
|   +-- rl_env/
|       +-- assets/          # Robot USD assets
|       +-- mdp/             # MDP components (rewards, actions, observations, ...)
|       +-- tasks/           # Task definitions (self-contained configs)
|       +-- rsl_rl/          # RSL-RL integration wrappers
+-- scripts/                 # train.py, eval.py, play.py, export, setup
+-- tests/                   # Unit and E2E tests
+-- workflows/               # Docker + OSMO remote training configs
+-- run.py                   # CLI for remote OSMO training/eval
+-- third_party/rsl_rl/      # AGILE patch for public rsl-rl-lib 5.4.1
```
