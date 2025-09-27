# **AGILE**: **A** **G**eneric **I**saac-**L**ab based **E**ngine for humanoid loco-manipulation learning

## Overview

**AGILE** provides a comprehensive reinforcement learning framework for training whole-body control policies with validated sim-to-real transfer capabilities. Built on NVIDIA Isaac Lab, this toolkit enables researchers and practitioners to develop loco-manipulation behaviors for humanoid robots.

<p align="center">
  <!-- Row 1 -->
  <span style="display:inline-block; text-align:center; margin:0px;">
    <img src="docs/videos/booster_t1_stand_up_sim2sim.gif" width="240">
  </span>
  <span style="display:inline-block; text-align:center; margin:0px;">
    <img src="docs/videos/booster_t1_vel_sim2sim.gif" width="240">
  </span>
  <span style="display:inline-block; text-align:center; margin:0px;">
    <img src="docs/videos/booster_t1_vel_sim2real.gif" width="240">
  </span>
  <br>
  <!-- Row 2 -->
  <span style="display:inline-block; text-align:center; margin:0px;">
    <img src="docs/videos/unitree_g1_vel_height_sim2sim.gif" width="240">
  </span>
  <span style="display:inline-block; text-align:center; margin:0px;">
    <img src="docs/videos/unitree_g1_vel_height_sim2real.gif" width="240">
  </span>
  <span style="display:inline-block; text-align:center; margin:0px;">
    <img src="docs/videos/unitree_g1_teleop.gif" width="240">
  </span>
</p>

<p align="center">
  <em><strong>Top row:</strong> Booster T1 – stand-up recovery (sim-to-sim), velocity tracking (sim-to-sim), velocity tracking (sim-to-real).<br>
  <strong>Bottom row:</strong> Unitree G1 – velocity-height tracking (sim-to-sim), velocity-height tracking (sim-to-real), teleoperation with trained policy.</em>
</p>

<details>

<summary> Key Features </summary>

### Key Features

<p align="center">
  <img src="docs/figures/agile_highlights.png" alt="AGILE Highlights" width="90%">
</p>
<p align="center"><em>Figure: AGILE highlights and key features.</em></p>

</details>

<details>
<summary> Project Structure </summary>

### Project Structure

```
agile/                       # Repository root
├── agile/                   # Main package
│   ├── algorithms/          # Algorithms for policy training
│   │   ├── rsl_rl/          # Custom rsl_rl library with TensorDict support
│   │   └── evaluation/      # Evaluation and metrics computation
│   ├── data/                # Data handling and policy checkpoints
│   ├── isaaclab_extras/     # Isaac Lab extensions and monkey patches
│   └── rl_env/              # Reinforcement learning environments
│       ├── assets/          # Robot assets and configurations
│       ├── mdp/             # MDP components (rewards, commands, actions, etc.)
│       ├── tasks/           # Task definitions and configurations
│       ├── tests/           # Unit tests for MDP components
│       ├── utils/           # Environment utilities
│       └── rsl_rl/          # RSL-RL integration and wrappers
├── docs/                    # Documentation and media files
│   └── videos/              # Demo videos (tracked with Git LFS)
├── scripts/                 # Utility scripts
│   ├── train.py             # Training script
│   ├── eval.py              # Evaluation script
│   ├── play.py              # Interactive play script
│   ├── verify_rsl_rl.py     # Verify RSL-RL installation
│   ├── export_IODescriptors.py # Export I/O descriptors
│   ├── setup/               # Installation and setup scripts
│   │   ├── install_deps.sh           # Install for Docker deployment
│   │   ├── install_deps_ci.sh        # Install for CI environment
│   │   ├── install_deps_local.sh     # Install for local development
│   │   └── setup_hooks.sh            # Set up git hooks
│   ├── wandb_sweep/         # Hyperparameter optimization with W&B
├── tests/                   # Test suite
├── workflows/               # Support workflow such as docker file
├── pyproject.toml           # Project configuration
├── CONTRIBUTING.md          # Contribution guidelines
└── README.md                # Project documentation
```
</details>

## Table of Contents
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Local Development Setup](#local-development-setup)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Embodiments](#embodiments)
  - [Tasks & Policy Architecture](#tasks--policy-architecture)
  - [Training](#training)
  - [Teacher Student Distillation](#teacher-student-distillation)
  - [Hyperparameter Sweep](#hyperparameter-sweep)
  - [Evaluation](#evaluation)
  - [Play](#play)
  - [Testing](#testing)
- [Development](#development)
  - [Docker Build Process](#docker-build-process)
  - [Pre-commit Hooks](#pre-commit-hooks)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Core Contributors](#core-contributors)
- [Acknowledgments](#acknowledgments)


## Installation

<details open>
<summary> Prerequisites </summary>

### Prerequisites

**Install Isaac Lab 2.3.0**:
Follow the installation [guide](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html). Note that Isaac Sim 5.1 is required to use the verified USD provided in this project.
We recommend using the conda installation. Remember to check out the specific branch as follows.
   ```bash
   # Ensure you're using version 2.3.0
   git checkout v2.3.0
   ```
</details>

<details>
<summary> Local Development Setup </summary>

### Local Development Setup

For local development on your machine:

```bash
# Ensure ISAACLAB_PATH is set
export ISAACLAB_PATH=/path/to/isaac_lab

# Install all dependencies and packages
./scripts/setup/install_deps_local.sh

# Verify the custom rsl_rl is correctly installed
${ISAACLAB_PATH}/isaaclab.sh -p scripts/verify_rsl_rl.py
```

The `scripts/setup/install_deps_local.sh` script will:
- Install runtime dependencies (tensordict, wandb, datasets, etc.)
- Remove any conflicting rsl_rl packages from Isaac Lab
- Install our custom rsl_rl with TensorDict support
- Install the agile package
</details>

## Quick Start

<details open>
<summary> Getting Started </summary>

Get started with AGILE in three simple steps locally:

**1. Train a velocity tracking policy:**

```bash
python scripts/train.py \
    --task Velocity-T1-v0 \
    --num_envs 2048 \
    --headless
```

**2. Visualize the trained policy:**

```bash
# After training completes, visualize the policy
python scripts/play.py \
    --task Velocity-T1-v0 \
    --num_envs 32 \
    --checkpoint <path_to_checkpoint>
```
> We also provide several trained policy in the data/policy folder.


> **💡 Next Steps:**
> - Explore [available tasks](#tasks--policy-architecture) for different robots and behaviors
> - Learn about [teacher-student distillation](#teacher-student-distillation) for robust deployment
> - See [Evaluation](#evaluation) for performance analysis and metrics

</details>

## Usage

<details>
<summary> Embodiments </summary>

The framework has been validated on two humanoid robots: Booster T1 and Unitree G1, with both robot USDs available in Isaac Sim 5.1 public release. For the G1 robot, we provide two actuator configurations: a delayed DC motor model and an implicit actuator setup adapted from [BeyondMimic](https://github.com/Beyond-Mimic/BeyondMimic), both verified in sim-to-sim and sim-to-real transfers.

</details>

<details open>
<summary> Tasks & Policy Architecture </summary>

### Tasks & Policy Architecture

#### Modular Policy Design

AGILE uses a modular approach to enable complex loco-manipulation behaviors:

<p align="center">
  <img src="docs/figures/separate_upper_lower_body_policy_diagram.png" alt="Modular Policy Architecture" width="80%">
</p>

The framework separates **lower body locomotion** (trained via RL) from **upper body control** (IK/IL/Random), with optional distillation to deployable student policies. This architecture enables flexible behavior composition and efficient training strategies.

**🎯 Teleoperation Integration**: AGILE policies power [Isaac Lab's official teleoperation examples](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/teleop_imitation.html#teleoperation). For optimal performance, use the latest policies from this repository—Isaac Lab will be updated with these improved versions soon.


> **Note**: This modular architecture represents our current implementation focus for loco-manipulation tasks, particularly enabling **teleoperation** where the upper body responds to external commands while maintaining stable locomotion. AGILE is not limited to this approach—the framework supports various policy architectures including unified full-body control (e.g., stand-up task) and will expand to support additional architectures in future releases.

#### Self-Contained Task Design

Each task configuration is **intentionally self-contained** with all MDP components in one file:

- ✅ **Transparent & Maintainable**: Complete setup visible without inheritance tracing
- ✅ **Collaboration-Friendly**: Developers work independently without conflicts
- ✅ **Fast Iteration**: Localized changes with immediate, visible impact

#### Available Tasks

This project supports multiple tasks across different robot embodiments (G1 and T1):

- **Locomotion**: Velocity tracking for G1 (legs + waist) and T1 (legs only)
- **Locomotion + Height**: Extended tracking with height commands, includes teacher and student distillation variants (recurrent & history-based)
- **Stand Up**: Full-body autonomous recovery from arbitrary fallen poses

> 📖 **For detailed task specifications, MDP configurations, complete design philosophy, and training pipeline documentation, see the [Task README](agile/rl_env/tasks/README.md).**
>
> 💡 **We've included [Lessons Learned](LESSONS_LEARNED.md) to share practical insights and tips from our experience developing these policies—from robot modeling to sim-to-real deployment.**



</details>

<details>
<summary> Training </summary>

### Training

Following Isaac Lab conventions, most training configuration lives in the corresponding `rsl_rl_ppo_cfg.py` file. Many options can be overridden via CLI. Run for full help:
```bash
python scripts/train.py -h
```

For local training, use the following command. We use W&B for logging by default.
```bash
python scripts/train.py \
    --task Velocity-T1-v0 \
    --num_envs 4096 \
    --headless \
    --logger wandb \
    --log_project_name Velocity-T1-v0 \
    --run_name test
```

> **💡 Experiment Reproducibility:** Training (including evaluation) automatically captures and logs lightweight git metadata (commit hash, branch, uncommitted changes, and diffs) to your experiment logs. When using W&B, this information is uploaded to your run for easy tracking and reproduction. This ensures you can always trace back the exact code state—including any staged or unstaged changes—used for any experiment, without storing the entire repository.


</details>

<details>
<summary> Teacher Student Distillation </summary>

### Teacher Student Distillation

**Teacher Training**
Training a teacher policy with privileged observations is often more effective than directly training a deployable policy using noisy and partially observable inputs. To train a teacher policy, follow the standard training procedure, adding any useful observations and removing noise. Once training is complete, export the policy using the play script.

**Student Distillation**
After obtaining the exported teacher policy (`.pt` file), you can distill it into a student policy that uses realistic (i.e., deployable) observations.

To configure the distillation process, set up the runner as follows:

```python
@configclass
class DistillationRunnerCfg(TeacherPpoRunnerCfg):
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        loss_type="mse",
    )
    policy = RslRlStudentTrainedTeacherCfg(
        class_name="StudentTrainedTeacher",  # or "StudentTrainedTeacherRecurrent"
        teacher_path="/path/to/exported/teacher_policy.pt",
        student_hidden_dims=[256, 256, 128],
        activation="elu",
    )
```

In the environment configuration, define separate observation dictionaries:
- `policy`: for student observations
- `teacher`: for teacher observations (this corresponds to the `critic` in RL training). This is simply what you defined as `policy` observations during teacher training.

Finally, register the task as a standard `rsl_rl` task and start training. Note that during distillation, the reward is not used for optimization—it is still logged for reference.

**Tip**
Training the student as a recurrent network is often beneficial as it can help to cope with noise and partial observability.
</details>

<details>
<summary> Hyperparameter Sweep </summary>

### Hyperparameter Sweep

Deploy a W&B sweep for hyperparameter optimization, see [scripts/wandb_sweep/README](scripts/wandb_sweep/README.md) for details.
</details>

<details>
<summary> Evaluation </summary>

### Evaluation

Evaluate trained policies with deterministic scenarios and report generation:

```bash
python scripts/eval.py \
    --task Velocity-Height-G1-v0 \
    --checkpoint /path/to/model.pt \
    --num_envs 1024 \
    --headless
```

Additional evaluation options include `--save_trajectories` to save trajectory data for analysis, `--generate_report` to generate HTML evaluation reports, `--eval_config` to use deterministic evaluation scenarios, and more. Run with `--run_evaluation` to enable the full evaluation pipeline. See [agile/algorithms/evaluation/README.md](agile/algorithms/evaluation/README.md) for detailed configurations.
</details>

<details>
<summary> Play </summary>

### Play
To play and export a trained policy, use the `scripts/play.py` script, e.g.,:
```bash
python scripts/play.py --num_envs 32 --task Velocity-Height-G1-v0 --resume RESUME --load_run 2025-01-01_00-00-0_task_name
```
Use the `resume` and `load_run` flags to select the run to export. This saves the exported policy (ONNX and TorchScript) in the `exported/` directory. You can also provide a `checkpoint` path directly.
</details>

<details>
<summary> Testing </summary>

### Testing

```bash
# Run all tests in Docker (matches CI environment)
./tests/test_e2e_ci_locally.sh --all

# Run locally (requires Isaac Lab)
./tests/run_unit_tests.sh
```

See [tests/README.md](tests/README.md) for detailed testing guide.
</details>


## Development

<details>
<summary> Docker Build Process </summary>

### Docker Build Process

The `workflows/Dockerfile`:
1. Starts from `nvcr.io/nvidia/isaac-lab:2.3.0` base image
2. Installs Python dependencies into Isaac Lab's environment
3. Removes conflicting rsl_rl packages
4. Installs custom rsl_rl with TensorDict support
5. Verifies correct installation
</details>

<details>
<summary> Pre-commit Hooks </summary>

### Pre-commit Hooks

This repository uses pre-commit hooks to ensure code quality. To set up the hooks:

1. Install the pre-commit hooks:
```bash
./scripts/setup/setup_hooks.sh
```

2. The hooks will run automatically on each commit. To run them manually:
```bash
pre-commit run --all-files
```

The pre-commit configuration includes:
- Code formatting with Black and isort
- Linting with Flake8
- Type checking with mypy
- Various file checks (trailing whitespace, merge conflicts, etc.)

Note: The `third_party` directory is excluded from all pre-commit hooks to preserve the original code style of external dependencies.
</details>

## Deployment
Policy deployment for both sim-to-sim and sim-to-real transfer currently utilizes NVIDIA's internal deployment framework, which is planned for public release in the near future.

**Pre-trained Policies:** We include several verified pre-trained checkpoints in the repository for evaluation and deployment. See [`agile/data/policy/README.md`](agile/data/policy/README.md) for available policies and usage instructions.

## Troubleshooting

<details>
<summary> Common issues </summary>

**Issue: `ModuleNotFoundError: No module named 'tensordict'`**
- The dependencies are not installed in Isaac Lab's Python environment
- Solution: Re-run `./scripts/setup/install_deps_local.sh` for local development or rebuild Docker image with `--rebuild`

**Issue: Wrong rsl_rl version being used**
- Isaac Lab's bundled rsl_rl is taking precedence
- Solution: Run `${ISAACLAB_PATH}/isaaclab.sh -p scripts/verify_rsl_rl.py` to check which version is installed
- The custom version should show TensorDict support

**Issue: Docker build fails at verification step**
- The custom rsl_rl was not properly installed
- Check that `agile/algorithms/rsl_rl/` exists and contains the custom implementation

**Issue: Isaac Sim initialization failures in containers**
- The wrapper automatically retries failed training runs (2 attempts with 10s delay)
- This handles common Isaac Sim cold start issues in Docker containers

</details>


## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed information on how to contribute to this project.

## License

This repository contains code under two different open-source licenses:

### BSD 3-Clause License
The reinforcement learning algorithm library located in `agile/algorithms/rsl_rl/` is licensed under the **BSD 3-Clause License**.
- **Copyright holders:** ETH Zurich, NVIDIA CORPORATION & AFFILIATES
- This portion is based on the [RSL_RL](https://github.com/leggedrobotics/rsl_rl) library developed at ETH Zurich
- See the full BSD 3-Clause license text in the [LICENCE](LICENCE) file (Section A)

### Apache License 2.0
All other portions of this repository are licensed under the **Apache License 2.0**.
- **Copyright holder:** NVIDIA CORPORATION & AFFILIATES
- See the full Apache 2.0 license text in the [LICENCE](LICENCE) file (Section B)

### Compliance
When using or distributing this software, you must comply with both licenses as applicable:
- If you modify or redistribute the `agile/algorithms/rsl_rl/` directory, comply with the BSD 3-Clause License terms
- For all other code, comply with the Apache 2.0 License terms

For complete license information and full terms, see the [LICENCE](LICENCE) file at the root of this repository.

## Core Contributors
Huihua Zhao, Rafael Cathomen, Lionel Gulich, Efe Arda Ongan, Michael Lin, Shalin Jain, Wei Liu, Vishal Kulkarni, Soha Pouya, Yan Chang

## Acknowledgments
We would like to acknowledge the following projects from which parts of the code in this repo are derived:
- [Beyond Mimic](https://github.com/HybridRobotics/whole_body_tracking)
- [RSL_RL](https://github.com/leggedrobotics/rsl_rl)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)

## Citation
If you use AGILE in your research, please cite:

```bibtex
@misc{agile2025,
  title        = {AGILE: A Generic Isaac-Lab based Engine for Humanoid Loco-Manipulation Learning},
  author       = {Zhao, Huihua and Cathomen, Rafael and Gulich, Lionel and Ongan, Efe Arda and Lin, Michael and Jain, Shalin and Liu, Wei and Kulkarni, Vishal and Pouya, Soha and Chang, Yan},
  year         = {2025},
  note         = {Version compatible with Isaac Lab 2.3; accessed 2025-11-19},
  url          = {https://github.com/nvidia-isaac/WBC_AGILE/tree/main}
}
```
