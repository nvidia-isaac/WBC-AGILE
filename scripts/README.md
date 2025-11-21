# Scripts

This directory contains all scripts for training, evaluation, and system management for the AGILE project.

## Core Scripts

### `train.py`
Main training script for RL agents using RSL-RL. Supports distributed training, video recording, and W&B logging. Use this to train new policies from scratch or resume training from checkpoints.

Example usage:
```bash
python scripts/train.py --task Velocity-T1-v0 --num_envs 4096 --video --video_interval_iter 500 --headless --logger wandb
```

### `eval.py`
Evaluation script for trained RL agents. Loads checkpoints, evaluates agent performance, and automatically exports policies to TorchScript and ONNX formats. Used for quantitative analysis, benchmarking, and policy export for deployment.

Example usage:
```bash
python scripts/eval.py --task Velocity-T1-v0 --num_envs 32 --checkpoint path/to/checkpoint.pt
```

### `play.py`
Environment validation script that runs environments with sinusoidal test actions (no policy required). Useful for debugging environment configurations, testing new robots, and validating MDP components.

Example usage:
```bash
python scripts/play.py --task Velocity-T1-v0 --num_envs 32
```

### `sim2mujoco_eval.py`
Evaluation script for Sim2Sim transfer to MuJoCo. Runs trained policies in MuJoCo simulation to verify transfer performance before real hardware deployment. This is a **generic framework** that works with any task by automatically parsing the I/O descriptor YAML file.

Example usage:
```bash
python scripts/sim2mujoco_eval.py --checkpoint path/to/policy.pt --config path/to/config.yaml --mjcf path/to/robot.xml
```

**Quick Start Tutorial:**

1. **Export Policy to TorchScript:**
   ```bash
   python scripts/eval.py --task Velocity-G1-History-v0 --checkpoint path/to/checkpoint.pt
   # This automatically exports policy.pt in the checkpoint directory's exported/ folder
   ```

2. **Export I/O Descriptors:**
   ```bash
   python scripts/export_IODescriptors.py --task Velocity-G1-History-v0 --output_dir path/to/output
   # Generates a YAML file describing observation/action spaces
   ```

3. **Get Robot MJCF:**
   We recommend using official robot models from [Unitree's MuJoCo repository](https://github.com/unitreerobotics/unitree_mujoco):
   ```bash
   git clone https://github.com/unitreerobotics/unitree_mujoco.git
   # G1 robot: unitree_mujoco/unitree_robots/g1/g1_29dof.xml
   ```

4. **Run Sim2MuJoCo Evaluation:**
   ```bash
   python scripts/sim2mujoco_eval.py \
     --checkpoint path/to/policy.pt \
     --config path/to/config.yaml \
     --mjcf unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
     --duration 10.0
   ```

> **💡 Interactive Control:** The sim2mujoco module supports keyboard teleoperation. Use arrow keys (↑↓←→) or I/J/K/L for movement, U/O for turning, and Page Up/Down (or 9/0) for height control. Press SPACE to stop. Remove `--no-viewer` flag to enable the interactive viewer.

## Utility Scripts


### `export_IODescriptors.py`
Exports I/O descriptors from the environment configuration. Generates a yaml file describing observation and action spaces for the trained models. Can be used for deployment in isaac-deploy

Example usage:
```bash
python scripts/export_IODescriptors.py --task Velocity-T1-v0 --output_dir .
```

### `extract_git_info.sh`
Bash script to extract git repository information (commit hash, branch, uncommitted changes) before Docker builds for reproducibility tracking.

## Subdirectories

### `setup/`
Shell scripts for environment setup and dependency management:
- `install_deps.sh` - Install Python dependencies and configure Isaac Sim environment
- `install_deps_ci.sh` - CI-specific dependency installation with optimizations for automated builds
- `install_deps_local.sh` - Local development setup including additional developer tools
- `setup_hooks.sh` - Configure git hooks for code quality checks (pre-commit, formatting)

### `wandb_sweep/`
Hyperparameter optimization using Weights & Biases:
- `init_sweep.py` - Initialize new hyperparameter sweeps
- `run_sweep.py` - Run sweep agents for distributed hyperparameter search
- `train_wrapper.py` - Wrapper script that processes sweep parameters
- `sweep.yaml` - Sweep configuration defining parameter search spaces
- See [wandb_sweep/README.md](scripts/wandb_sweep/README.md) for detailed usage instructions
