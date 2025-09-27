# RL Tasks

This directory contains all reinforcement learning task environments. Tasks are organized by behavior type and robot model.

## Folder Structure

```
tasks/
├── <task_category>/          # e.g., locomotion, stand_up, debug
│   ├── __init__.py           # Imports robot-specific modules
│   └── <robot>/              # e.g., g1, t1
│       ├── __init__.py       # Registers gym environments (task IDs)
│       ├── *_env_cfg.py      # Environment configuration(s)
│       ├── agents/           # RL algorithm configurations
│       │   ├── __init__.py
│       │   └── rsl_rl_ppo_cfg.py  # PPO hyperparameters
```

**Key Components:**
- **Task ID Registration**: Each `<robot>/__init__.py` registers gym environments with unique task IDs (e.g., `Velocity-T1-v0`)
- **Environment Config**: Defines scene, robot, observations, rewards, terminations, and curriculum
- **Agent Config**: Specifies RL algorithm hyperparameters (network architecture, learning rates, etc.)

## Training Command

To train a policy:

```bash
python scripts/train.py --task <TASK_ID>
```

**Example:**
```bash
python scripts/train.py --task Velocity-T1-v0
```

## Available Tasks

### 1. Locomotion (`locomotion/`)

Velocity tracking tasks where robots learn to follow commanded linear and angular velocities.

#### Unitree G1 Robot

| Task ID | Description |
|---------|-------------|
| `Velocity-G1-History-v0` | Velocity tracking with observation history stacking |

#### Booster T1 Robot

| Task ID | Description |
|---------|-------------|
| `Velocity-T1-v0` | Standard velocity tracking task for T1 robot |

> **Note**: The G1 and T1 locomotion environments share nearly identical MDP configurations, differing only in robot-specific joints and links. This design philosophy minimizes embodiment-specific tuning and demonstrates that the same training setup can potentially be applied to other robot platforms with minimal modifications. To achieve this uniformity, G1's waist roll and pitch joints are included in the lower body controller while the yaw joint remains uncontrolled, matching T1's degrees of freedom of the waist.

---

### 2. Locomotion with Height Commands (`locomotion_height/`)

Extended velocity tracking tasks that include height tracking.

#### Unitree G1 Robot

| Task ID | Description |
|---------|-------------|
| `Velocity-Height-G1-v0` | Velocity tracking with height commands |
| `Velocity-Height-G1-Distillation-Recurrent-v0` | Student policy with recurrent network |
| `Velocity-Height-G1-Distillation-History-v0` | Student policy with history stacking |

---

### 3. Stand Up (`stand_up/`)

Tasks for learning to stand-up.

#### Booster T1 Robot

| Task ID | Description |
|---------|-------------|
| `StandUp-T1-v0` | Stand-up task for T1 robot |

---

### 4. Debug (`debug/`)

Special environment for debugging MDP terms (rewards, observations), symmetry functions, and robot models. This environment launches two floating robots with a GUI for direct joint control, allowing users to change target positions and PD gains of all joints interactively.

The two robots are present to verify symmetry functions: actions are mirrored through the symmetry functions between the two robots. The GUI is implemented as an action, so rewards and observations run normally during debugging.

| Task ID | Description |
|---------|-------------|
| `Debug-G1-v0` | Debug environment for Unitree G1 |
| `Debug-T1-v0` | Debug environment for Booster T1 |

## Adding New Tasks

To create a new task:

1. **Create task directory structure:**
   ```
   tasks/<new_task_category>/<robot>/
   ├── __init__.py
   ├── <task_name>_env_cfg.py
   └── agents/
       ├── __init__.py
       └── rsl_rl_ppo_cfg.py
   ```

2. **Register the environment** in `<robot>/__init__.py`:
   ```python
   import gymnasium as gym
   from . import agents

   gym.register(
       id="<TaskName>-<Robot>-v0",
       entry_point="isaaclab.envs:ManagerBasedRLEnv",
       disable_env_checker=True,
       kwargs={
           "env_cfg_entry_point": f"{__name__}.<task_name>_env_cfg:<ConfigClassName>",
           "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:<RunnerConfigClassName>",
       },
   )
   ```

3. **Update parent `__init__.py`** files to import your new module

4. **Configure environment and agent** in the respective config files

---

## Notes

- **Distillation Tasks**: These tasks use teacher-student learning where a teacher policy (with privileged observations) trains a student policy (with partial observations like on hardware). To use distillation tasks, you must specify the path to the teacher policy file in the PPO config file. The teacher must be a JIT-exported policy.

For more details on environment configuration, see the individual `*_env_cfg.py` files in each task directory.
