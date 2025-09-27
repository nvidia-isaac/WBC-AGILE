# WandB Sweep System for Isaac Lab

This directory contains a WandB (Weights & Biases) [sweep](https://docs.wandb.ai/models/sweeps) system for hyperparameter optimization in Isaac Lab.


## Key Features

- **WandB Sweep Integration**: Sweeping and logging with WandB
- **Scaled-Dictionary Parameters**: Special support for scaling complex robot configurations (actuator gains, effort limits, etc.)
- **Multi-Machine Support**: Run agents across multiple machines/GPUs that coordinate through WandB via sweep ID.

## Files Overview

- `sweep.yaml` - Main sweep configuration file. To define once
- `train_wrapper.py` - Wrapper script that processes sweep parameters
- `init_sweep.py` - Initialize a new sweep. To run once
- `run_sweep.py` - Start sweep agents. To run as many times as desired on different machines
- `sweep_ids.json` - Stores sweep IDs for reuse across machines

## Quick Start

### 1. Configure Your Sweep

Edit `sweep.yaml` to define your hyperparameter sweep:

```yaml
method: bayes
entity: YOUR_WANDB_ENTITY  # Replace with your WandB username
metric:
  name: Metrics/base_velocity/error_vel_xy
  goal: minimize

command:
  - python
  - scripts/wandb_sweep/train_wrapper.py
  - --num_envs
  - 64
  - --task
  - Isaac-Velocity-G1-Lower-Agile-v0
  - --logger
  - wandb
  - --headless

parameters:
  agent.algorithm.learning_rate:
    distribution: log_uniform_values
    min: 1.0e-4
    max: 1.0e-3

  # Scaled-dictionary parameters
  p_gain_leg:
    distribution: uniform
    min: 0.5
    max: 2.0
    p_gain_leg_cli_path:
      value: env.scene.robot.actuators.legs.stiffness
    p_gain_leg_base_dict:
      value: |
        {
          ".*_hip_yaw_joint": 100.0,
          ".*_hip_roll_joint": 100.0,
          ".*_hip_pitch_joint": 100.0,
          ".*_knee_joint": 200.0,
          "waist.*": 200.0
        }
```

### 2. Initialize the Sweep

```bash
python scripts/wandb_sweep/init_sweep.py --project_name my_robot_optimization
```
This stores the newly generated sweep id in the `sweep_ids.json` file.

### 3. Start Sweep Agents

```bash
# Single agent that will run 1 experiment then stop
python scripts/wandb_sweep/run_sweep.py --project_name my_robot_optimization --agent_count 1

# Single agent that will run 4 experiments sequentially then stop
python scripts/wandb_sweep/run_sweep.py --project_name my_robot_optimization --agent_count 4
```

**Important**: The `--agent_count` parameter specifies how many experiments a single agent will run **sequentially**, not in parallel.

### For Parallel Optimization

To run multiple experiments in parallel, you need to launch multiple instances of the script:

```bash
# On multiple machines or terminals:
# Terminal 1:
python scripts/wandb_sweep/run_sweep.py --project_name my_robot_optimization --agent_count 10

# Terminal 2:
python scripts/wandb_sweep/run_sweep.py --project_name my_robot_optimization --agent_count 10

# Repeat same command as many times as desired.

```

## Scaled-Dictionary Parameters

This system supports a special pattern for scaling complex dictionary parameters, which is particularly useful for robot actuator configurations.

### Pattern

For each scaled-dictionary parameter, define the parameter with nested configuration:

1. Distribution and range (min/max)
2. `param_name_cli_path` - The Hydra path to the parameter
3. `param_name_base_dict` - The baseline dictionary values

### Example: Actuator Stiffness Scaling

```yaml
parameters:
  p_gain_leg:
    distribution: uniform
    min: 0.5    # 50% of baseline values
    max: 2.0    # 200% of baseline values
    # Nested configuration for clarity
    p_gain_leg_cli_path:
      value: env.scene.robot.actuators.legs.stiffness
    p_gain_leg_base_dict:
      value: |
        {
          ".*_hip_yaw_joint": 100.0,
          ".*_hip_roll_joint": 100.0,
          ".*_hip_pitch_joint": 100.0,
          ".*_knee_joint": 200.0,
          "waist.*": 200.0
        }
```

This will generate commands like:
```bash
env.scene.robot.actuators.legs.stiffness={.*_hip_yaw_joint:150.0,.*_hip_roll_joint:150.0,.*_hip_pitch_joint:150.0,.*_knee_joint:300.0,waist.*:300.0}
```



## Usage Commands

### Initialize a New Sweep
```bash
python scripts/wandb_sweep/init_sweep.py --project_name PROJECT_NAME
```

### Run Sweep Agents
```bash
python scripts/wandb_sweep/run_sweep.py --project_name PROJECT_NAME --agent_count N
```

### Monitor Progress
View results in WandB dashboard: `https://wandb.ai/ENTITY/PROJECT_NAME`
