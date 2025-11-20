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
