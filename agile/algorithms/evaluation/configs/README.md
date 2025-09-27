# Evaluation Scenario Configurations

This directory contains YAML configuration files for deterministic evaluation scenarios. These configs allow you to test specific capabilities systematically by controlling commands, terrain, and other environment parameters.

## Quick Start

```bash
# Run with a specific evaluation scenario
python scripts/eval.py \
    --task Velocity-Height-G1-Dev-v0 \
    --checkpoint path/to/model.pt \
    --eval_config agile/algorithms/evaluation/configs/examples/x_velocity_sweep.yaml \
    --num_envs 1
```

## Configuration Format

```yaml
evaluation:
  task_name: "Velocity-Height-G1-Dev-v0"  # Task to evaluate
  num_envs: 4                              # Number of parallel environments
  episode_length_s: 50.0                   # Episode duration (longer than training)
  num_episodes: 1                          # Number of episodes to run

  environments:
    - env_ids: [0]                         # Which environment IDs this applies to
      name: "x_velocity_test"              # Descriptive name for this test

      # Option 1: Use sweep (shorthand for uniform intervals)
      sweep:
        interval: 5.0                      # Change command every 5 seconds
        commands:
          base_velocity:
            lin_vel_x: [-1.0, 0.0, 1.0]   # Cycles through these values
            lin_vel_y: 0.0                 # Fixed values
            ang_vel_z: 0.0
            base_height: 0.75

      # Option 2: Use explicit schedule (full control)
      schedule:
        - time: 0.0
          commands:
            base_velocity:
              lin_vel_x: 0.5
              lin_vel_y: 0.0
              ang_vel_z: 0.0
              base_height: 0.75
        - time: 10.0
          commands:
            base_velocity:
              lin_vel_x: 1.0
              # ... must specify all fields
```

## Available Examples

### Single Environment Tests

1. **x_velocity_sweep.yaml** - Tests forward/backward walking
   - Sweeps through lin_vel_x: [-1.0, -0.5, 0.0, 0.5, 1.0]
   - Changes every 5 seconds

2. **y_velocity_sweep.yaml** - Tests lateral walking
   - Sweeps through lin_vel_y: [-1.0, -0.5, 0.0, 0.5, 1.0]

3. **yaw_rate_sweep.yaml** - Tests turning
   - Sweeps through ang_vel_z: [-1.0, -0.5, 0.0, 0.5, 1.0]

4. **height_sweep.yaml** - Tests different standing heights
   - Sweeps through base_height: [0.55, 0.60, 0.65, 0.70, 0.75]

### Multi-Environment Tests

5. **multi_env_capability_test.yaml** - Tests all capabilities in parallel
   - Env 0: X-velocity sweep
   - Env 1: Y-velocity sweep
   - Env 2: Yaw rate sweep
   - Env 3: Height sweep

### Advanced Examples

6. **explicit_schedule_example.yaml** - Complex maneuver sequence
   - Stand → Walk forward → Turn → Strafe → Backward+turn → Stand
   - Demonstrates explicit time-based scheduling

## Creating Custom Scenarios

### 1. Command Requirements

All `base_velocity` commands must specify all 4 fields:
```yaml
commands:
  base_velocity:
    lin_vel_x: 0.5    # Required
    lin_vel_y: 0.0    # Required
    ang_vel_z: 0.0    # Required
    base_height: 0.75 # Required
```

Commands are automatically clamped to valid ranges defined in the task config.

### 2. Using Sweep vs Schedule

**Use `sweep`** for:
- Uniform time intervals
- Simple parameter sweeps
- Less verbose configs

**Use `schedule`** for:
- Non-uniform timing
- Complex sequences
- Different parameters at different times

**Use both** for:
- Sweep + occasional special events
- Most flexibility

### 3. Multi-Environment Design

Assign different tests to different environments:

```yaml
environments:
  - env_ids: [0, 1]      # Envs 0 and 1 run the same test
    name: "test_a"
    sweep: ...

  - env_ids: [2]         # Env 2 runs different test
    name: "test_b"
    schedule: ...

  - env_ids: [3]         # Env 3 yet another test
    name: "test_c"
    sweep: ...
```

Unassigned environments will use random commands (training behavior).

## Future Extensions

The config system is designed to support additional overrides:

```yaml
schedule:
  - time: 10.0
    terrain:                # FUTURE: Change terrain difficulty
      terrain_level: 2

    events:                 # FUTURE: Trigger disturbances
      enable: ["push_robot"]
      push_robot:
        force_range: [-20.0, 20.0]

    physics:                # FUTURE: Modify physics
      friction: 0.3
```

## Tips

1. **Start with single env**: Test configs with `num_envs: 1` first
2. **Check ranges**: Commands are clamped to task config ranges
3. **Episode length**: Use longer episodes than training (e.g., 50s vs 30s)
4. **Combine with logging**: Use `--save_trajectories` to analyze results
5. **Validate configs**: Scheduler prints summary at startup - check it!

## Example Workflows

**Test specific velocity:**
```bash
python scripts/eval.py --task MyTask-v0 --checkpoint model.pt \
    --eval_config configs/examples/x_velocity_sweep.yaml --save_trajectories
```

**Full capability evaluation:**
```bash
python scripts/eval.py --task MyTask-v0 --checkpoint model.pt \
    --eval_config configs/examples/multi_env_capability_test.yaml \
    --num_envs 4 --save_trajectories
```

**Custom scenario:**
```bash
# Create my_test.yaml
python scripts/eval.py --task MyTask-v0 --checkpoint model.pt \
    --eval_config my_test.yaml
```
