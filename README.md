# **AGILE**: **A** **G**eneric **I**saac-**L**ab based **E**ngine for humanoid loco-manipulation learning

## Overview

AGILE provides a comprehensive reinforcement learning framework for training whole-body control policies with validated sim-to-real transfer capabilities. Built on NVIDIA Isaac Lab, this toolkit enables researchers and practitioners to develop loco-manipulation behaviors for humanoid robots.

<details open>

<summary> Key Features </summary>

### Key Features

**🤖 Robot Support & Validated Tasks**
- **Multi-robot embodiments**: Unitree G1 and Booster T1
- **Multi tasks**: Different task settings, including velocity tracking, height tracking and standing up to provide comprehensive examples for environment setup.
- **Sim2Real validated**: Proven transfer for both G1 and T1 robots in real-world deployment


**🛠️ Development & Debugging Tools**
- **Debug environment**: Rapid prototyping task to verify joint configurations, rewards, symmetry and robot setup
- **Isaac Lab manager-based architecture**: Modular environment design for easy customization
- **Extensive MDP library**: Delayed actuator models, terrain generation, [random action generator](agile/rl_env/mdp/actions/velocity_profiles/README.md), reward functions, and more
- **IO export**: Exporting the whole task setup as a yaml file for quick deployment, see [scripts/README.md](agile/scripts/README.md)

**📊 Training Infrastructure**
- **Enhanced RSL-RL**: Extended with TensorDict support, entropy annealing, symmetry losses, and teacher-student distillation (see [MODIFICATIONS](agile/algorithms/rsl_rl/MODIFICATIONS.md))
- **WandB integration**: Hyperparameter sweeps and experiment tracking with automatic Git commit logging.
- **Adaptive curriculum**: Harness force simulation with progressive difficulty
- **Teacher-student distillation**: Train robust policies with privileged information, then distill to deployable student policies

**📈 Evaluation & Analysis**
- **Deterministic evaluation**: Evaluation wrapper to allow to run the environment deterministically
- **Trajectory saving**: Export data for detailed offline analysis
- **Automated report generation**: HTML reports with performance metrics and trajectory analysis

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
│   └── sys_id/              # System identification scripts
├── tests/                   # Test suite
├── workflows/               # Support workflow such as docker file and remote cluster training.
├── pyproject.toml           # Project configuration
├── CONTRIBUTING.md          # Contribution guidelines
└── README.md                # Project documentation
```
</details>

<!-- ### Roadmap

Future releases will expand AGILE's capabilities with additional behaviors and features:
- **Extended behaviors**: Squatting with Sim2Real, torso RPY tracking, system identification
- **Geometric fabrics**: Physics-informed locomotion control
- **Manipulation policies**: Whole-body loco-manipulation tasks -->

## Demos

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


## Installation

<details open>
<summary> Prerequisites </summary>

### Prerequisites

**Install Isaac Lab 2.3.0**:
Follow the installation [guide](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html). Note that, Isaac Sim 5.1 is required to use the verified USD provided in this project.
We recommend using the conda installation. Remember to checkout to the specific branch shown as the following.
   ```bash
   # Ensure you're using version 2.3.0
   git checkout v2.3.0
   export ISAACLAB_PATH=/path/to/isaac_lab
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

## Usage

<details>
<summary> Embodiments </summary>

The framework has been validated on two humanoid robots: Booster T1 and Unitree G1, with both robot USDs available in Isaac Sim 5.1 public release. For the G1 robot, we provide two actuator configurations: a delayed DC motor model and an implicit actuator setup adapted from [BeyondMimic](https://github.com/Beyond-Mimic/BeyondMimic), both verified in sim-to-sim and sim-to-real transfers.

</details>

<details>
<summary> Tasks </summary>

### Tasks

This project supports multiple tasks across different robot embodiments (G1 and T1). For detailed task descriptions and configurations, see the task [README](agile/rl_env/tasks/README.md).
</details>

<details>
<summary> Training </summary>

### Training

Following the convention of Isaac Lab, most of the training configuration is in the corresponding `rsl_rl_ppo_cfg.py` file. It is still possible to overwrite some of them. Run the following command for details.
```py
python scripts/train.py -h
```

For local training, it can be started with the following command. We use wandb for logging by default.
```py
python scripts/train.py \
    --task Velocity-T1-v0 \
    --num_envs 4096 \
    --headless \
    --logger wandb \
    --log_project_name Velocity-T1-v0 \
    --run_name test
```
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
Use the `resume` and `load_run` flag to select the run to export. This will save the exported policy (onnx and torch.jit) in the `exported` directory.
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


## Deployment
Policy deployment for both sim-to-sim and sim-to-real transfer currently utilizes NVIDIA's internal deployment framework, which is planned for public release in the near future.

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
- Type checking with MyPy
- Various file checks (trailing whitespace, merge conflicts, etc.)

Note: The `third_party` directory is excluded from all pre-commit hooks to preserve the original code style of external dependencies.
</details>

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

This project is licensed under the NVIDIA license - see the license headers in source files for details.
