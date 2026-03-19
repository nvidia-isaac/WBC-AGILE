# Evaluation

AGILE provides two evaluation paths -- **Isaac Lab** and **Sim2MuJoCo** -- that follow the same
design principle. Both paths share an identical workflow: load a trained policy, apply commands
(deterministic schedules, sweeps, or random), roll out the policy in simulation, and save
trajectory data for analysis. They use the same YAML eval config format, produce the same
Parquet output schema, and work with the same plotting and analysis tools.

The two paths differ in simulator backend and feature set:

| Aspect | Isaac Lab | Sim2MuJoCo |
|--------|-----------|------------|
| Script | `scripts/eval.py` | `scripts/sim2mujoco_eval.py` |
| Simulator | Isaac Sim (GPU) | MuJoCo (CPU) |
| Parallel envs | Yes (N envs) | Single env |
| Eval config | Shared YAML format | Shared YAML format |
| Output format | Parquet + metadata.json | Parquet + metadata.json |
| Metrics (metrics.json) | Yes | No |
| HTML reports | Yes | No |
| Interactive control | No | Keyboard teleop |
| Random commands | Yes | Yes |
| Observation noise | Yes | Yes |

## Isaac Lab Evaluation

```bash
# Evaluate a trained policy
python scripts/eval.py \
    --task Velocity-T1-v0 \
    --num_envs 32 \
    --checkpoint /path/to/model.pt \
    --run_evaluation

# With trajectory saving and HTML report
python scripts/eval.py \
    --task Velocity-T1-v0 \
    --num_envs 32 \
    --checkpoint /path/to/model.pt \
    --run_evaluation \
    --save_trajectories \
    --generate_report

# With a deterministic evaluation scenario
python scripts/eval.py \
    --task Velocity-Height-G1-v0 \
    --num_envs 16 \
    --checkpoint /path/to/model.pt \
    --run_evaluation \
    --eval_config agile/algorithms/evaluation/configs/examples/x_velocity_sweep.yaml
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--run_evaluation` | Enable PolicyEvaluator |
| `--save_trajectories` | Save trajectory data to parquet files |
| `--trajectory_fields` | Specific fields to save (default: all) |
| `--generate_report` | Generate HTML report (requires `--save_trajectories`) |
| `--eval_config` | Path to YAML scenario config for deterministic testing |

### Output Structure

```
logs/rsl_rl/<experiment_name>/
  trajectories/
    episode_000.parquet
    episode_001.parquet
    ...
  metrics.json
  reports/          # if --generate_report
    index.html
    episodes/
      episode_000.html
      ...
```

## Sim2MuJoCo Evaluation

The Sim2MuJoCo path runs policies in MuJoCo for cross-simulator validation. See
{doc}`sim2mujoco` for setup instructions (policy export, MJCF acquisition).

```bash
# Interactive keyboard control
python scripts/sim2mujoco_eval.py \
    --checkpoint /path/to/policy.pt \
    --config /path/to/config.yaml \
    --mjcf /path/to/scene.xml \
    --duration 30.0

# Deterministic evaluation (same YAML config format as Isaac Lab)
python scripts/sim2mujoco_eval.py \
    --checkpoint /path/to/policy.pt \
    --config /path/to/config.yaml \
    --mjcf /path/to/scene.xml \
    --eval-config agile/sim2mujoco/configs/x_velocity_sweep.yaml \
    --save-data --no-viewer

# Random commands (reproducible with seed)
python scripts/sim2mujoco_eval.py \
    --checkpoint /path/to/policy.pt \
    --config /path/to/config.yaml \
    --mjcf /path/to/scene.xml \
    --random-commands all --random-interval 2.0 --random-seed 42 \
    --duration 50.0 --save-data --no-viewer
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--checkpoint` | Path to policy checkpoint (`.pt` or `.onnx`) |
| `--config` | Path to exported I/O descriptor YAML |
| `--mjcf` | Path to MuJoCo MJCF file (overrides config default) |
| `--duration` | Simulation duration in seconds |
| `--eval-config` | Path to YAML eval config (deterministic command schedule) |
| `--save-data` | Save trajectory data to Parquet files |
| `--output-dir` | Custom output directory for saved data |
| `--random-commands` | Randomize commands: field names (`vx`, `vy`, `wz`, `height`) or `all` |
| `--random-interval` | Seconds between random resamples (default: 2.0) |
| `--random-seed` | RNG seed for reproducible random commands |
| `--noise-scale` | Observation noise scale (0=off, 1=match training, >1=stress test) |
| `--pd-scale` | Scale factor for PD gains (use 0.3--0.5 for stability) |
| `--no-viewer` | Disable MuJoCo viewer (headless mode) |
| `--no-real-time` | Disable real-time pacing (runs as fast as possible) |

### Command Modes

Three mutually exclusive command modes are available:

- **Keyboard control** (default): Interactive teleoperation via the MuJoCo viewer.
  Arrow keys for movement, U/O for turning, Page Up/Down for height.
- **Eval config** (`--eval-config`): Deterministic command schedules from YAML files,
  using the same format as Isaac Lab evaluation. Duration is set from the config's
  `episode_length_s`.
- **Random commands** (`--random-commands`): Uniform random resampling at a fixed interval.
  Specify individual fields (`vx`, `vy`, `wz`, `height`) or `all`. Use `--random-seed`
  for reproducibility.

```{note}
`--eval-config` and `--random-commands` are mutually exclusive. Keyboard control is
automatically disabled when either is active or when `--no-viewer` is set.
```

### Output Structure

```
logs/sim2mujoco/<task>/<eval_config>_<timestamp>/
  trajectories/
    metadata.json
    episode_000.parquet
```

The Parquet schema matches the Isaac Lab output: `joint_pos_{i}`, `joint_vel_{i}`,
`joint_acc_{i}`, `root_pos_{i}`, `root_lin_vel_robot_{i}`, `commands_{i}`, `actions_{i}`,
plus metadata columns (`episode_id`, `env_id`, `frame_idx`, `timestep`).

## Deterministic Scenario Configs

Both evaluation paths use the same YAML config format for deterministic testing. Configs
define controlled command sequences instead of random commands, enabling systematic and
reproducible evaluation.

Isaac Lab configs live in `agile/algorithms/evaluation/configs/examples/`;
Sim2MuJoCo configs live in `agile/sim2mujoco/configs/`. The format is identical --
only task-specific values (sweep ranges, durations) differ.

Two specification modes are available:

### Sweep Mode

Uniform time intervals cycling through a list of values:

```yaml
evaluation:
  task_name: "Velocity-Height-G1-Dev-v0"
  num_envs: 4
  episode_length_s: 50.0
  num_episodes: 1

  environments:
    - env_ids: [0]
      name: "x_velocity_test"
      sweep:
        interval: 5.0
        commands:
          base_velocity:
            lin_vel_x: [-1.0, 0.0, 1.0]
            lin_vel_y: 0.0
            ang_vel_z: 0.0
            base_height: 0.75
```

### Schedule Mode

Explicit time-based command sequences for complex maneuvers:

```yaml
environments:
  - env_ids: [0]
    name: "complex_maneuver"
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
            lin_vel_y: 0.0
            ang_vel_z: 0.0
            base_height: 0.75
```

### Multi-Environment Testing

Assign different tests to different environments (Isaac Lab only -- Sim2MuJoCo runs a single env):

```yaml
environments:
  - env_ids: [0, 1]
    name: "test_a"
    sweep: ...

  - env_ids: [2]
    name: "test_b"
    schedule: ...
```

Unassigned environments use random commands (training behavior).

### Pre-built Scenarios

Isaac Lab examples in `agile/algorithms/evaluation/configs/examples/`:

| Config | Description |
|--------|-------------|
| `x_velocity_sweep.yaml` | Forward/backward walking |
| `y_velocity_sweep.yaml` | Lateral movement |
| `yaw_rate_sweep.yaml` | Turning |
| `height_sweep.yaml` | Height control |
| `multi_env_capability_test.yaml` | All capabilities in parallel (one per env) |
| `explicit_schedule_example.yaml` | Complex maneuver sequence |

Sim2MuJoCo examples in `agile/sim2mujoco/configs/`:

| Config | Description |
|--------|-------------|
| `x_velocity_sweep.yaml` | Forward/backward velocity sweep |
| `y_velocity_sweep.yaml` | Lateral velocity sweep |
| `yaw_rate_sweep.yaml` | Turning rate sweep |
| `height_sweep.yaml` | Base height sweep (velocity+height tasks) |

All `base_velocity` commands must specify all 4 fields (`lin_vel_x`, `lin_vel_y`, `ang_vel_z`, `base_height`). Commands are automatically clamped to valid ranges defined in the task config.

```{tip}
Start with `num_envs: 1` to validate configs. Use longer episodes than training (e.g., 50s vs 30s) for thorough testing.
```

## HTML Reports

Interactive HTML reports with tracking analysis and per-joint plots. Reports are generated
by the Isaac Lab evaluation path. The Sim2MuJoCo path does not generate reports directly,
but its Parquet output is compatible with the plotting API for custom analysis (see
Analyzing Trajectories below).

### Generation

```bash
# Automatic (during evaluation)
python scripts/eval.py --task <task_name> --checkpoint path/to/model.pt \
    --run_evaluation --save_trajectories --generate_report

# Manual (after evaluation)
python agile/algorithms/evaluation/generate_report.py \
    --log_dir logs/evaluation/task_datetime

# Specific or failed episodes only
python agile/algorithms/evaluation/generate_report.py \
    --log_dir logs/evaluation/task_datetime \
    --episodes failed

# Specific episode IDs
python agile/algorithms/evaluation/generate_report.py \
    --log_dir logs/evaluation/task_datetime \
    --episodes 0,3,5
```

### Report Contents

- **Summary Dashboard** (`index.html`): Success rate, sortable episode table with search/filter, tracking error summary plots
- **Episode Pages** (`episodes/episode_XXX.html`): Tracking performance (lin_vel_x, lin_vel_y, ang_vel_z, height), all joints organized by body part (upper/lower) with collapsible sections, joint position and velocity limits shown, interactive Plotly plots (zoom, pan, hover)

## Analyzing Trajectories (Python/Jupyter)

The plotting API works with trajectory data from both evaluation paths, since they share the
same Parquet schema and metadata format.

```python
import sys
sys.path.insert(0, "agile/algorithms/evaluation")
from plotting import load_episode, load_metadata, plot_joint_trajectories
import matplotlib.pyplot as plt

# Works with either Isaac Lab or Sim2MuJoCo output directories
metadata = load_metadata("logs/rsl_rl/experiment")
df = load_episode("logs/rsl_rl/experiment", episode_id=0)

fig, axes = plot_joint_trajectories(
    df,
    joint_names=['left_hip_yaw_joint', 'right_knee_joint'],
    metadata=metadata,
    show_limits=True,
)
plt.show()
```

## Evaluation Framework Internals

The evaluation framework lives in `agile/algorithms/evaluation/`.

### PolicyEvaluator

The main evaluation class (`evaluator.py`) that collects trajectory data from policy rollouts and computes metrics:

- Requires an `eval` observation group providing joint positions, velocities, accelerations, root state, commands, and actions
- Handles terminal state observations correctly by using previous-frame data for terminated environments
- Supports configurable joint groups for per-body-part metrics
- Optionally saves trajectory data to Parquet files for offline analysis

### MotionMetricsAnalyzer

Computes and aggregates motion quality metrics (`motion_metrics_analyzer.py`):

- **Mean/max joint acceleration**: Smoothness indicator (lower is better)
- **Mean/max acceleration rate (jerk)**: Jerkiness indicator
- **Mean/max joint velocity**: Activity level
- All metrics computed for whole body and per joint group
- Separate statistics for all episodes vs. successful episodes only
- Results saved as JSON with grouped metrics

### TrajectoryReportGenerator

Generates interactive HTML reports from saved trajectory data (`report_generator.py`):

- Uses Plotly for interactive, zoomable plots
- Supports filtering by success/failure status
- Works standalone without Isaac Sim (only requires pandas, plotly, jinja2)
