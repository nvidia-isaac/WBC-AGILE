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

---

### 💡 Design Philosophy: Self-Contained Task Configurations

Each task configuration file (`*_env_cfg.py`) is intentionally **self-contained** with all MDP components (scene, observations, actions, rewards, terminations, events, curriculum) defined in one place.

**Why this approach?**

This design choice deviates from Isaac Lab's inheritance-based approach based on practical experience:

- ❌ **Deep inheritance obscures setup**: Multi-layer hierarchies make it difficult to track the exact MDP configuration
- ❌ **Child modifications break parents**: Changes in derived classes can inadvertently introduce subtle bugs in base configurations
- ❌ **Task-specific tuning is inevitable**: Every task requires fine-tuning, making shared base classes less valuable in practice

**Benefits in practice:**

- ✅ **Transparency & Maintainability**: Complete configuration visible in one file, no inheritance tracing needed
- ✅ **Seamless with Isaac Lab**: Works naturally with Isaac Lab's manager-based environment architecture
- ✅ **Efficient Collaboration**: Multiple developers can work on different tasks independently without conflicts
- ✅ **Faster Iteration**: Changes are localized with immediate, visible impact—no hidden side effects

> 🎯 **Bottom line**: We prioritize transparency and maintainability over code reuse, making it easier to understand, debug, and modify individual tasks.

---

## Separated Lower and Upper Body Policy Architecture

To enable complex whole-body behaviors like locomotion with manipulation, AGILE uses a modular policy design that separates lower body locomotion from upper body control. This architecture allows for flexible composition of behaviors and efficient training strategies.

<p align="center">
  <img src="../../../docs/figures/separate_upper_lower_body_policy_diagram.png" alt="Modular Policy Architecture" width="100%">
</p>

#### Training Pipeline

#### Step 1: Teacher Policy Training

- **Lower Body (Locomotion)**: Trained end-to-end using reinforcement learning (RL) to track velocity, height, and other locomotion commands. The locomotion policy receives observations from the robot's state and environment, and outputs joint position targets for the lower body.
- **Upper Body (Manipulation)**: Can use different approaches depending on the task:
  - **Inverse Kinematics (IK)**: For simple position-based control
  - **Imitation Learning (IL)**: For more complex manipulation behaviors
  - **Random policies**: For training robust locomotion that is agnostic to upper body movements

The teacher policy typically has access to privileged information (e.g., ground truth terrain height, friction coefficients) that may not be available on the real robot, making it powerful but not directly deployable.

#### Step 2: Student Policy Distillation (Optional)

- After training the teacher, we can distill its behavior into a deployable student policy that only uses realistic observations available on hardware (e.g., joint positions, velocities, IMU readings).
- Two student architectures are supported:
  - **Recurrent networks (LSTM/GRU)**: Better at handling noise and partial observability through temporal memory
  - **MLP with history stacking**: Simpler architecture that concatenates recent observation history
- The distillation step can be skipped if the teacher policy is already deployable (i.e., it doesn't rely on privileged information).

#### Benefits

This modular design enables:

- **Independent development** of locomotion and manipulation behaviors
- **Reusable locomotion policies** across different upper body tasks
- **Robust training** with privileged information, followed by efficient distillation to deployable policies
- **Flexibility** to adapt to different robot morphologies and task requirements

> **Note on Architectural Flexibility**: AGILE supports multiple policy architectures beyond the modular approach described above. The separated lower-upper body design provides greater flexibility for tasks requiring accurate teleoperation and manipulation behaviors, where the upper body needs to respond to external commands (e.g., IK targets from a teleop interface) while maintaining stable locomotion. However, other tasks such as the **stand-up task** control the full body joints in a unified manner, as they require coordinated whole-body movements where separating upper and lower body control is less beneficial. The choice of architecture depends on the specific requirements of your application.

---

## Available Tasks

### 1. Locomotion (`locomotion/`)

Velocity tracking tasks where robots learn to follow commanded linear and angular velocities.

#### Unitree G1 Robot

| Task ID | Controlled Joints | Policy Type | Commands | Observations | Actuator Model |
|---------|-------------------|-------------|----------|--------------|----------------|
| `Velocity-G1-History-v0` | Legs + Waist Roll/Pitch (14 joints) | Teacher (Non-Privileged) | Velocity (x, y, yaw) | History (5 steps) | Delayed DC Motor |

#### Booster T1 Robot

| Task ID | Controlled Joints | Policy Type | Commands | Observations | Actuator Model |
|---------|-------------------|-------------|----------|--------------|----------------|
| `Velocity-T1-v0` | Legs only (12 joints) | Teacher (Non-Privileged) | Velocity (x, y, yaw) | History (5 steps) | Delayed DC Motor |

> **Note**: The G1 and T1 locomotion environments share nearly identical MDP configurations, differing only in robot-specific joints and links. This design philosophy minimizes embodiment-specific tuning and demonstrates that the same training setup can potentially be applied to other robot platforms with minimal modifications. To achieve this uniformity, G1's waist roll and pitch joints are included in the lower body controller while the yaw joint remains uncontrolled, matching T1's degrees of freedom of the waist.

---

### 2. Locomotion with Height Commands (`locomotion_height/`)

Extended velocity tracking tasks that include height tracking. The teacher policy uses privileged information (terrain height scans), while student policies are distilled for deployment.

#### Unitree G1 Robot

| Task ID | Controlled Joints | Policy Type | Commands | Observations | Actuator Model |
|---------|-------------------|-------------|----------|--------------|----------------|
| `Velocity-Height-G1-v0` | Legs only (12 joints) | Teacher (Privileged) | Velocity (x, y, yaw) + Height | No history | Delayed Implicit Actuator |
| `Velocity-Height-G1-Distillation-Recurrent-v0` | Legs only (12 joints) | Student (Recurrent LSTM) | Velocity (x, y, yaw) + Height | Recurrent | Delayed Implicit Actuator |
| `Velocity-Height-G1-Distillation-History-v0` | Legs only (12 joints) | Student (History stacking) | Velocity (x, y, yaw) + Height | History (5 steps) | Delayed Implicit Actuator |

> **Note on Waist Joint Control:** These G1 tasks purposefully control only the leg joints while leaving the waist joint uncontrolled by the policy. This design choice provides the upper body IK policy more freedom to expand the workspace while maintaining accuracy. By decoupling lower and upper body control, this approach benefits both the robustness of the lower body learned policy and the accuracy of the upper body IK controller, delivering an improved teleoperation experience.

---

### 3. Stand Up (`stand_up/`)

Tasks for learning to recover from arbitrary fallen poses and stand up. Uses full-body control with a lifting assistance mechanism during training.

#### Booster T1 Robot

| Task ID | Controlled Joints | Policy Type | Commands | Observations | Actuator Model |
|---------|-------------------|-------------|----------|--------------|----------------|
| `StandUp-T1-v0` | Full body (all joints) | Teacher (Non-Privileged) | None | History (5 steps) | Delayed DC Motor |

**Pre-collected Fallen States:**

The stand-up task uses a pre-collected dataset of fallen robot states for efficient episode resets. Instead of simulating the robot falling at the start of each episode (which takes ~2 seconds), states are sampled from the dataset for instant resets.

- **Automatic collection**: The `pre_learn` hook automatically collects/loads fallen states before training starts
- **Caching**: Collected states are cached to disk and reused across training runs (cache invalidates when terrain config changes)
- **Validation**: Use `python scripts/play.py --task StandUp-T1-v0 --validate-fallen-states` to visualize the collected poses

Configuration in `agents/rsl_rl_ppo_cfg.py`:
```python
fallen_state_dataset_cfg: FallenStateDatasetCfg = FallenStateDatasetCfg(
    num_spawns_per_level=2,  # States collected per terrain level
    fall_duration_s=2.5,      # Simulation time for falling
    cache_enabled=True,       # Enable disk caching
)
```

---

### 4. Pick and Place (`pick_place/`)

Motion tracking tasks for learning pick-and-place manipulation behaviors. The robot learns to track reference trajectories for the upper body while maintaining stable lower body control via a pre-trained locomotion policy.

#### Unitree G1 Robot

| Task ID | Controlled Joints | Policy Type | Commands | Observations | Actuator Model |
|---------|-------------------|-------------|----------|--------------|----------------|
| `G1-PickPlace-Tracking-v0` | Right arm + hand + waist (16 joints) | Teacher (Non-Privileged) | Trajectory tracking | Current state | Implicit Actuator |
| `G1-PickPlace-Tracking-v0-Debug` | Right arm + hand + waist (16 joints) | Evaluation | Trajectory tracking | Current state | Implicit Actuator (GUI-tunable PD gains) |

**Key Features:**
- **Modular Policy Architecture**: Upper body (manipulation) and lower body (locomotion) are controlled by separate policies
- **Reference Trajectory Tracking**: Upper body tracks pre-recorded motion trajectories from YAML files
- **Pre-trained Locomotion**: Lower body uses a frozen pre-trained locomotion policy for stability
- **Object Interaction**: Scene includes a table and manipulable object for pick-and-place tasks

**Debug Environment Features:**
- Interactive joint position GUI for the right arm/hand
- Object pose GUI for manipulating object position and rotation
- Reward visualizer for monitoring reward terms in real-time
- Contact sensor visualization on fingertips

To run:
```bash
# Training
python scripts/train.py --task G1-PickPlace-Tracking-v0

# Play trained policy
python scripts/eval.py --task G1-PickPlace-Tracking-v0 --checkpoint path/to/checkpoint
```

---

### 5. Debug (`debug/`)

Special environments for debugging MDP terms (rewards, observations), symmetry functions, and robot models. These environments provide interactive GUI controls for direct joint and object manipulation.

#### Joint Debug Environment

Launches two floating robots with a GUI for direct joint control, allowing users to change target positions and PD gains of all joints interactively. The two robots verify symmetry functions: actions are mirrored through the symmetry functions between them.

![Joint Debug GUI](../../../docs/videos/debug_gui_g1.gif)

| Task ID | Description |
|---------|-------------|
| `Debug-G1-v0` | Joint debug environment for Unitree G1 |
| `Debug-T1-v0` | Joint debug environment for Booster T1 |

#### Object Interaction Debug Environment

Extends the joint debug environment with interactive object pose control. Useful for debugging robot-object interaction policies by manually positioning both the robot joints and the object.

![Object Interaction Debug](../../../docs/videos/debug_robot_object_interaction.gif)

| Task ID | Description |
|---------|-------------|
| `Debug-G1-Object-v0` | Object interaction debug environment for Unitree G1 |

**Features:**
- Joint position GUI control for the robot
- Object pose GUI control (position + rotation)
- Reward visualizer to monitor reward terms in real-time
- Configurable position/rotation limits for the object

To run the debug environments:
```bash
# Joint debug
python scripts/play.py --task Debug-G1-v0 --num_envs 2

# Object interaction debug
python scripts/play.py --task Debug-G1-Object-v0
```

**Tip:** To adapt an existing task for GUI debugging, see the usage example in `debug/object_debug_env_cfg.py`.

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
