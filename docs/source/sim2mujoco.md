# Sim-to-MuJoCo Transfer

AGILE can run LEAPP-exported policies in MuJoCo for cross-simulator validation.
The Sim2MuJoCo runner uses LEAPP's Python `InferenceManager` to reload the
exported policy graph, maps LEAPP semantic inputs to MuJoCo state, and applies
LEAPP joint target/gain outputs to the MuJoCo robot.

For an overview of LEAPP bundles and policy export, see
{doc}`deploy-policies`. This page uses the exported bundle with AGILE's Python
MuJoCo validation runner.

## Quick Start

1. **Export the policy through LEAPP**:

```bash
uv run scripts/export_policy_leapp.py \
    --task Velocity-Height-G1-History-v0 \
    --checkpoint /path/to/model.pt
```

2. **Download the public robot assets** or bring your own MJCF:

```bash
uv run agile-download-assets
# G1 robot: external_assets/unitree_mujoco/unitree_robots/g1/scene_29dof.xml
```

3. **Run the LEAPP bundle in MuJoCo**:

```bash
uv run scripts/sim2mujoco_eval.py \
    --leapp-yaml logs/rsl_rl/<experiment>/<run>/Velocity-Height-G1-History-v0/Velocity-Height-G1-History-v0.yaml \
    --mjcf external_assets/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
    --duration 10.0
```

`--leapp-yaml` points to the LEAPP YAML file in the exported bundle. The bundle is
self-contained: the policy joints' PD gains ride in the exported graph and the control
frequency is recorded under `pipeline.configs.frequency`. The runner derives everything else
from the bundle and the MJCF: the control decimation from the MJCF physics timestep, the reset
pose from zeros, joint armature/limits from the MJCF, and default gains for any joint the policy
does not control. No companion file is required.

```{tip}
If the robot is unstable in MuJoCo, try `--pd-scale 0.3` to reduce PD gains.
```

## Interactive Control

The Sim2MuJoCo module supports keyboard teleoperation. Remove `--no-viewer` to
enable the interactive viewer:

- Arrow keys (or I/J/K/L) for movement
- U/O for turning
- Page Up/Down (or 9/0) for height control
- SPACE to stop

## Deterministic Evaluation

For reproducible evaluations, use YAML-driven command schedules. These reuse the
same eval config format as the Isaac Lab evaluation pipeline:

```bash
uv run scripts/sim2mujoco_eval.py \
    --leapp-yaml logs/rsl_rl/<experiment>/<run>/Velocity-Height-G1-History-v0/Velocity-Height-G1-History-v0.yaml \
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

Use `--save-data` to record per-step data to parquet files:

```bash
uv run scripts/sim2mujoco_eval.py \
    --leapp-yaml logs/rsl_rl/<experiment>/<run>/Velocity-Height-G1-History-v0/Velocity-Height-G1-History-v0.yaml \
    --mjcf /path/to/scene.xml \
    --save-data --output-dir logs/sim2mujoco/my_eval
```

Output structure:

```text
logs/sim2mujoco/<task>/<eval>_<timestamp>/
  trajectories/
    episode_000.parquet
  metadata.json
```
