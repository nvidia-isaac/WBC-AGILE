# Evaluation Module

This module provides policy evaluation tools for humanoid locomotion tasks:

- **PolicyEvaluator**: Computes motion smoothness metrics and success rate
- **Trajectory logging**: Save complete episode data for offline analysis
- **Deterministic scenarios**: Test specific capabilities with controlled commands

## Command Line Usage

**A trained checkpoint is required.** Run evaluations using `eval.py`:

### Basic Evaluation

```bash
python scripts/eval.py --task <task_name> --checkpoint path/to/model.pt \
    --num_envs 16 --run_evaluation
```

### Evaluation with Trajectory Logging

Save complete episode data for offline analysis:

```bash
python scripts/eval.py --task <task_name> --checkpoint path/to/model.pt \
    --num_envs 16 --run_evaluation --save_trajectories
```

**Output:**
```
logs/rsl_rl/<experiment_name>/
├── trajectories/
│   ├── episode_000.parquet
│   ├── episode_001.parquet
│   └── ...
└── metrics.json
```

### Deterministic Scenario Evaluation

Run controlled tests with specific commands instead of random ones:

```bash
python scripts/eval.py --task <task_name> --checkpoint path/to/model.pt \
    --eval_config agile/algorithms/evaluation/configs/examples/x_velocity_sweep.yaml \
    --run_evaluation --save_trajectories --generate_report
```

**Benefits:**
- Reproducible testing
- Systematic capability evaluation
- Per-environment control
- Time-based command changes
- Automatic HTML report generation

**Example scenarios** (in `configs/examples/`):
- `x_velocity_sweep.yaml` - Test forward/backward walking
- `y_velocity_sweep.yaml` - Test lateral movement
- `yaw_rate_sweep.yaml` - Test turning
- `height_sweep.yaml` - Test height control
- `multi_env_capability_test.yaml` - Test all capabilities in parallel

See [configs/README.md](configs/README.md) for full documentation.

**How it works:**

```text
eval.py → Load YAML → Create scheduler → Evaluation loop: scheduler.update() → env.step()
```

The `VelocityHeightScheduler` applies time-based overrides to velocity+height commands (lin_vel_x, lin_vel_y, ang_vel_z, base_height) per environment. Optional and backward-compatible - no config means default random testing. Currently supports velocity+height commands only; other command types will require a new scheduler implementation.

### Key Options

- `--task`: Task name (required)
- `--checkpoint`: Path to checkpoint (or auto-detects latest from logs)
- `--run_evaluation`: Enable PolicyEvaluator
- `--save_trajectories`: Save trajectory data to parquet files
- `--trajectory_fields`: Specific fields to save (default: all)
- `--num_envs`: Number of parallel environments (default: 16)
- `--eval_config`: Path to YAML scenario config (optional, for deterministic testing)
- `--generate_report`: Automatically generate HTML report after evaluation (requires `--save_trajectories`)

## Generating HTML Reports

**Interactive HTML reports** with all joints and tracking analysis:

### Automatic (During Evaluation)

```bash
python scripts/eval.py --task <task_name> --checkpoint path/to/model.pt \
    --run_evaluation --save_trajectories --generate_report
```

### Manual (After Evaluation)

```bash
# Generate report for all episodes
python agile/algorithms/evaluation/generate_report.py \
    --log_dir logs/evaluation/Velocity-Height-G1-Dev-v0_20251010_214925

# Generate for specific episodes
python agile/algorithms/evaluation/generate_report.py \
    --log_dir logs/evaluation/task_datetime \
    --episodes 0,3,5

# Generate for failed episodes only
python agile/algorithms/evaluation/generate_report.py \
    --log_dir logs/evaluation/task_datetime \
    --episodes failed
```

**Features:**
- **Summary Dashboard** (`index.html`):
  - Success rate and overall statistics
  - Sortable episode table (click columns to sort)
  - Search/filter episodes
  - Tracking error summary plots

- **Detailed Episode Pages** (`episodes/episode_XXX.html`):
  - Tracking performance (lin_vel_x, lin_vel_y, ang_vel_z, height)
  - All joints organized by body part (upper/lower)
  - Collapsible sections (click to expand/collapse)
  - Joint position and velocity limits shown
  - Interactive plotly plots (zoom, pan, hover)

**Output location:** `logs/evaluation/task_datetime/reports/`


<details>
<summary><b>📊 View Example Report Screenshots</b></summary>

<br>

**Summary Dashboard:**

![Evaluation Report Summary](../../../docs/figures/evaluation_report_summary.png)

**Detailed Tracking Analysis:**

![Evaluation Report Tracking](../../../docs/figures/evaluation_report_tracking.png)

</details>

## Analyzing Trajectories (Python/Jupyter)

```python
import sys
sys.path.insert(0, "agile/algorithms/evaluation")
from plotting import load_episode, load_metadata, plot_joint_trajectories
import matplotlib.pyplot as plt

# Load metadata and data
metadata = load_metadata("logs/rsl_rl/experiment")
df = load_episode("logs/rsl_rl/experiment", episode_id=0)

# Plot by joint names
fig, axes = plot_joint_trajectories(
    df,
    joint_names=['left_hip_yaw_joint', 'right_knee_joint'],
    metadata=metadata,
    show_limits=True
)
plt.show()
```

## Tests

```bash
python -m pytest agile/algorithms/evaluation/tests/test_evaluator.py -v
```
