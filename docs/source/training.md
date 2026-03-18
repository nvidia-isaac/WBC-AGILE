# Training Guide

AGILE uses the same training script interface as Isaac Lab. Run `python scripts/train.py -h` for all available arguments.

## Training

```bash
python scripts/train.py --task Velocity-T1-v0 --num_envs 4096 --headless --logger wandb
```

## Resuming Training

```bash
python scripts/train.py \
    --task Velocity-T1-v0 \
    --num_envs 4096 \
    --headless \
    --resume True \
    --load_run "2024-01-15_.*" \
    --checkpoint "model_5000.pt"
```

## Hydra Overrides

Environment and agent parameters can be overridden via Hydra-style arguments:

```bash
python scripts/train.py \
    --task Velocity-T1-v0 \
    --num_envs 4096 \
    --headless \
    env.rewards.track_lin_vel_xy_exp.weight=2.0 \
    agent.algorithm.learning_rate=5e-4
```

## Environment Validation (Play)

Before training, validate your task setup (scene, actions, MDP functions) using sinusoidal test actions -- no trained policy needed:

```bash
python scripts/play.py --task Velocity-T1-v0 --num_envs 2
```

For tasks with pre-collected fallen states (e.g., StandUp-T1-v0), validate the dataset:
```bash
python scripts/play.py --task StandUp-T1-v0 --num_envs 16 --validate-fallen-states
```

```{seealso}
See {doc}`evaluation` for the full evaluation guide including deterministic scenarios, HTML reports, and trajectory analysis.
```

## Teacher-Student Distillation

The standard pipeline for sim-to-real: train a privileged teacher, then distill to a deployable student.

```bash
# 1. Train the teacher (with privileged observations)
python scripts/train.py \
    --task Velocity-Height-G1-v0 \
    --num_envs 4096 --headless --logger wandb

# 2. Export the teacher
python scripts/export_policy.py \
    --task Velocity-Height-G1-v0 \
    --checkpoint /path/to/teacher/model_30000.pt --headless

# 3. Train the student (distills from exported teacher)
python scripts/train.py \
    --task Velocity-Height-G1-Distillation-Recurrent-v0 \
    --num_envs 4096 --headless --logger wandb

# 4. Evaluate and export the student
python scripts/eval.py \
    --task Velocity-Height-G1-Distillation-Recurrent-v0 \
    --num_envs 32 \
    --checkpoint /path/to/student/model_5000.pt \
    --run_evaluation --save_trajectories --generate_report

python scripts/export_policy.py \
    --task Velocity-Height-G1-Distillation-Recurrent-v0 \
    --checkpoint /path/to/student/model_5000.pt --headless
```

The student task config references the exported teacher via `teacher_path` in the agent config. Students can be MLP or recurrent (LSTM/GRU).

## Hyperparameter Sweeps

AGILE includes a [W&B sweep](https://docs.wandb.ai/models/sweeps) system for hyperparameter optimization. Files are in `scripts/wandb_sweep/`.

### Quick Start

```bash
# 1. Configure sweep.yaml with your parameters

# 2. Initialize (run once)
python scripts/wandb_sweep/init_sweep.py --project_name my-sweep

# 3. Start agents (run on each GPU/machine)
python scripts/wandb_sweep/run_sweep.py --project_name my-sweep --agent_count 10
```

The `--agent_count` parameter specifies how many experiments a single agent will run **sequentially**, not in parallel. To run in parallel, launch multiple instances across terminals or machines.

### Scaled-Dictionary Parameters

The sweep system supports a special pattern for scaling complex dictionary parameters, which is useful for robot actuator configurations. For each scaled-dictionary parameter, define:

1. Distribution and range (min/max as a scalar multiplier)
2. `param_name_cli_path` -- the Hydra path to the parameter
3. `param_name_base_dict` -- the baseline dictionary values

```yaml
parameters:
  # Standard parameter
  agent.algorithm.learning_rate:
    distribution: log_uniform_values
    min: 1.0e-4
    max: 1.0e-3

  # Scaled-dictionary parameter: multiplies all values in the dict
  p_gain_leg:
    distribution: uniform
    min: 0.5    # 50% of baseline
    max: 2.0    # 200% of baseline
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

This generates Hydra override commands like:
```
env.scene.robot.actuators.legs.stiffness={.*_hip_yaw_joint:150.0,...}
```

## Policy Export

```bash
# Export to TorchScript (JIT)
python scripts/export_policy.py \
    --task Velocity-T1-v0 \
    --checkpoint /path/to/model.pt --headless

# Or export during evaluation
python scripts/eval.py \
    --task Velocity-T1-v0 \
    --checkpoint /path/to/model.pt \
    --export_io
```
