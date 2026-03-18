# Sim-to-MuJoCo Transfer

AGILE includes a generic, task-agnostic framework (`agile/sim2mujoco/`) for running trained policies in MuJoCo simulation. The framework automatically handles observation/action mapping by parsing the exported I/O descriptor YAML file -- no code changes needed for different tasks.

## Quick Start

1. **Export policy and I/O descriptors** from your trained checkpoint:

```bash
python scripts/eval.py --task Velocity-G1-History-v0 --checkpoint /path/to/checkpoint.pt
python scripts/export_IODescriptors.py --task Velocity-G1-History-v0 --output_dir /path/to/output
```

2. **Get robot MJCF** from [Unitree's official repository](https://github.com/unitreerobotics/unitree_mujoco) or bring your own:

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git
# G1 robot: unitree_mujoco/unitree_robots/g1/scene_29dof.xml
```

3. **Run evaluation in MuJoCo**:

```bash
python scripts/sim2mujoco_eval.py \
    --checkpoint /path/to/exported/policy.pt \
    --config /path/to/exported/config.yaml \
    --mjcf unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
    --duration 10.0
```

```{tip}
If the robot is unstable in MuJoCo, try `--pd-scale 0.3` to reduce PD gains.
```

## Interactive Control

The sim2mujoco module supports keyboard teleoperation. Remove `--no-viewer` to enable the interactive viewer:

- Arrow keys (or I/J/K/L) for movement
- U/O for turning
- Page Up/Down (or 9/0) for height control
- SPACE to stop

## Deterministic Evaluation

For reproducible evaluations, use YAML-driven command schedules. These reuse the same eval config format as the Isaac Lab evaluation pipeline:

```bash
python scripts/sim2mujoco_eval.py \
    --checkpoint /path/to/policy.pt \
    --config /path/to/config.yaml \
    --mjcf /path/to/scene.xml \
    --eval-config agile/sim2mujoco/configs/x_velocity_sweep.yaml \
    --save-data --no-viewer
```

Pre-built sweep configs in `agile/sim2mujoco/configs/`:

| Config | Description |
|--------|-------------|
| `x_velocity_sweep.yaml` | Forward/backward velocity sweep |
| `y_velocity_sweep.yaml` | Lateral velocity sweep |
| `yaw_rate_sweep.yaml` | Turning rate sweep |
| `height_sweep.yaml` | Base height sweep (velocity+height tasks) |

## Data Logging

Use `--save-data` to record per-step data (joint positions, velocities, accelerations, torques, commands, root state) to parquet files. Output is compatible with the existing `agile.algorithms.evaluation.plotting` utilities:

```bash
python scripts/sim2mujoco_eval.py \
    --checkpoint /path/to/policy.pt \
    --config /path/to/config.yaml \
    --mjcf /path/to/scene.xml \
    --save-data --output-dir logs/sim2mujoco/my_eval
```

Output structure:

```
logs/sim2mujoco/<task>/<eval>_<timestamp>/
  trajectories/
    episode_000.parquet
  metadata.json
```
