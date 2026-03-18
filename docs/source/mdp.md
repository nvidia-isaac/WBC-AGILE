# MDP Components

This section documents the Markov Decision Process (MDP) building blocks in AGILE. These
components are defined in `agile/rl_env/mdp/` and are composed by task configurations to
define complete training environments.

AGILE builds on Isaac Lab's manager-based architecture: each MDP component (rewards,
observations, actions, etc.) is a function or class registered with the corresponding
manager. The top-level `agile.rl_env.mdp` module re-exports both Isaac Lab's built-in
MDP terms and AGILE's custom additions, so task configs can import everything from a
single namespace:

```python
from agile.rl_env import mdp
```

## Rewards

Reward functions are the core training signal. AGILE organizes rewards into four modules
based on their purpose.

### Task Rewards (`rewards/task_rewards.py`)

Primary rewards that define the training objective for each task.

**Velocity tracking** -- the main locomotion rewards:

| Function | Description |
|----------|-------------|
| `track_lin_vel_xy_yaw_frame_exp_weighted_simplified` | Track linear velocity (x, y) in the yaw-aligned frame with exponential kernel. Higher commanded velocities receive higher weight. |
| `track_ang_vel_z_world_exp_weighted_simplified` | Track angular velocity (yaw) in world frame with magnitude-based weighting. |
| `track_base_height_exp_smooth` | Track commanded base height using exponential kernel on smoothed height signal. |
| `base_height_exp` | Track a fixed target base height with optional terrain sensor adjustment. |
| `vel_xy_in_threshold` | Binary reward: 1.0 if velocity tracking error is within threshold. |

**Height tracking**:

| Function | Description |
|----------|-------------|
| `track_base_height` | Track commanded height with exponential kernel, active only during stance. |
| `base_height_in_threshold` | Binary reward for height within threshold of command. |
| `height_reached` | Bonus reward when target height is reached within tolerance. |

**Stand-up specific**:

| Function | Description |
|----------|-------------|
| `standing_at_timeout` | Bonus reward when the episode times out AND the robot is standing above a minimum height. Encourages both standing up and staying standing. |

**Trajectory tracking** (pick-and-place):

| Function | Description |
|----------|-------------|
| `static_at_goal_exp` | Reward for being static (low velocities) during the final portion of a trajectory. Uses progress-based gating. |
| `nominal_posture_at_end_exp` | Reward matching the final frame's joint posture at trajectory end. |

**Gait rewards**:

| Function | Description |
|----------|-------------|
| `stand_still` | Penalize foot lift when velocity commands are near zero (stance mode). |
| `phase_contact` | Reward correct foot contact timing using gait phase from command term (XNOR logic). |

Most tracking rewards use the **exponential kernel** pattern:

```python
reward = torch.exp(-error / std**2)
```

This provides a smooth gradient that is 1.0 at zero error and decays toward 0.0.
The `std` parameter controls the tolerance width.

### Tracking Rewards (`rewards/tracking_rewards.py`)

Rewards for the pick-and-place trajectory tracking task. These work with the
`TrackingCommand` term.

| Function | Description |
|----------|-------------|
| `motion_global_anchor_position_error_exp` | Track reference anchor (base) position in world frame. |
| `motion_global_anchor_orientation_error_exp` | Track reference anchor orientation using quaternion error. |
| `motion_tracked_joint_pos_error_exp` | Track reference joint positions for tracked joints. |
| `motion_object_position_error_exp` | Track reference object position in world frame. |
| `motion_object_orientation_error_exp` | Track reference object orientation. |
| `hand_object_distance_tracking_exp` | Reward hand-object proximity with automatic phase detection. Detects the lift peak in the reference trajectory and decays the reward after placement. |

### Aesthetic Rewards (`rewards/aestetic_rewards.py`)

Style and quality-of-motion rewards that shape **how** the robot moves, not just whether
it achieves the objective.

**Body stability**:

| Class/Function | Description |
|----------------|-------------|
| `body_acc_l2` | Penalize body linear and angular accelerations using velocity history. Can target root or any specified link. |
| `body_ang_vel_l2` | Penalize angular velocity of a body/link (reduces shaking). |
| `flat_body_orientation_exp` | Reward flat body orientation via small xy-components of projected gravity. |

**Foot quality**:

| Function | Description |
|----------|-------------|
| `feet_roll_l2` | Penalize non-flat foot roll angles. |
| `feet_yaw_diff_l2` | Penalize yaw difference between left and right feet (reduced during turns). |
| `feet_yaw_mean_vs_base` | Penalize foot yaw relative to the base frame. |
| `feet_distance_from_ref` | Penalize lateral distance deviation from a reference spacing. |
| `feet_stumble` | Penalize high horizontal contact forces on feet. |
| `feet_slip` | Penalize horizontal foot velocity when in ground contact. |
| `foot_orientation_l1` | L1 penalty on foot roll, pitch, yaw with configurable weights. |
| `impact_velocity_l1` | Penalize large impact velocities at foot contact. |
| `jumping` | Penalize both feet leaving the ground simultaneously. |
| `equal_foot_force` | Reward even force distribution across both feet (1.0 = perfectly balanced). |

**Stance-mode rewards** (active when velocity command is zero):

| Function | Description |
|----------|-------------|
| `joint_deviation_exp_if_standing` | Penalize joint deviation from defaults, only when standing. |
| `moving_if_standing` | Penalize body motion when standing above height threshold. |
| `equal_foot_force_if_null_cmd` | Reward balanced foot forces during stance. |

### Regularization Rewards (`rewards/regularization_rewards.py`)

Penalties that encourage smooth, efficient actuation.

**Action smoothness**:

| Function | Description |
|----------|-------------|
| `action_rate_l2` | L2 penalty on action change between timesteps. |
| `action_rate_rate_l2` | L2 penalty on the second derivative of actions (jerk in action space). |
| `joint_deviation_l2` | L2 penalty on joint position deviation from defaults. |

**Energy and torque efficiency**:

| Function | Description |
|----------|-------------|
| `torque_limits` | Penalize torques exceeding a soft limit (configurable fraction of hardware limit). |
| `incoming_forces_penalty` | Penalize large internal joint wrench forces above a threshold. |
| `contact_forces_l2` | Penalize contact forces exceeding a threshold (L2 squared). |
| `relax_if_null_cmd` | Penalize torque magnitude during stance (zero-velocity commands). |
| `relax_if_null_cmd_exp` | Reward low torques during stance using exponential kernel with cached torque limits. |

### Reward Visualizer (`rewards/reward_visualizer.py`)

A real-time visualization tool for monitoring individual reward terms during evaluation.
Used by the debug and pick-and-place debug environments. Each reward term is displayed
as a bar chart that updates every simulation step.

## Actions

Action terms define the policy's output space and how it maps to joint commands. Located
in `agile/rl_env/mdp/actions/`.

### Joint Position Actions

**`JointPositionActionCfg`** (Isaac Lab built-in)
: Standard joint position action with configurable scale and offset. The policy outputs
  deltas around default joint positions.

**`DeltaJointPositionAction`** / `DeltaJointPositionActionCfg`
: Outputs delta joint positions that accumulate over time. Supports per-joint scaling,
  separate "steady" joints held at defaults, and optional joint limits. Used for
  manipulation tasks where incremental motion is more natural.

**`SmoothJointPositionAction`** / `SmoothJointPositionActionCfg`
: Wraps joint position actions with exponential moving average (EMA) smoothing.
  Configurable `ema_smoothing_param` (1.0 = no smoothing).

### Random Actions

**`RandomPositionAction`** / `RandomActionCfg`
: Generates random joint positions for upper-body joints during locomotion training.
  Supports configurable velocity profiles (EMA, linear, trapezoidal) for smooth
  transitions between random targets, and optional stance-mode behavior.

**`RandomJointPositionAction`** / `RandomJointPositionActionCfg`
: Alternative random action with curriculum support. Can gradually increase randomization
  range during training for progressive difficulty.

### Policy Actions

**`AgileBasedLowerBodyAction`** / `AgileLowerBodyActionCfg`
: Runs a pre-trained, frozen RL policy as an action term. Used in the pick-and-place task
  to provide stable locomotion while training the upper-body policy. Takes the path to a
  JIT-exported policy model and an observation group name.

### GUI Actions

**`JointPositionGUIAction`** / `JointPositionGUIActionCfg`
: Interactive GUI slider control for all joints. Supports mirroring between left/right
  sides and adjustable PD gains. Used in debug environments.

**`ObjectPoseGUIAction`** / `ObjectPoseGUIActionCfg`
: Interactive GUI control for object position and rotation. Used in object debug
  environments.

### Assistance Actions

**`HarnessAction`** / `HarnessActionCfg`
: Simulates a simplified harness by applying external forces and torques to prevent
  falling. Configurable stiffness, damping, and force/torque limits. Supports height
  commands for dynamic target height.

**`LiftAction`** / `LiftActionCfg`
: Applies upward forces to lift the robot, with configurable ramp-up timing. Used in
  stand-up training with a curriculum that gradually reduces the assistance. Supports
  delayed start (`start_lifting_time_s`) and ramped lifting (`lifting_duration_s`).

### Velocity Profiles

Located in `actions/velocity_profiles/`, these define how random upper-body actions
transition between targets. All profiles use fully vectorized batch operations and
support synchronized joint motion.

| Profile | Description | Key Parameters |
|---------|-------------|----------------|
| `EMAVelocityProfileCfg` | Exponential moving average: `pos = alpha * target + (1 - alpha) * current`. Smooth convergence to targets. | `ema_coefficient_range` |
| `LinearVelocityProfileCfg` | Constant velocity motion: `pos = initial + velocity * time`. Predictable, time-based control. | `velocity_range` (rad/s) |
| `TrapezoidalVelocityProfileCfg` | Three-phase motion (acceleration, cruise, deceleration). Physically realistic. | `acceleration_range`, `max_velocity_range`, `deceleration_range` |

Usage example:

```python
from agile.rl_env.mdp.actions.velocity_profiles import TrapezoidalVelocityProfileCfg
from agile.rl_env.mdp.actions import RandomActionCfg

action_cfg = RandomActionCfg(
    asset_name="robot",
    joint_names=["joint1", "joint2"],
    sample_range=(0.1, 1.5),
    velocity_profile_cfg=TrapezoidalVelocityProfileCfg(
        acceleration_range=(1.0, 3.0),
        max_velocity_range=(0.5, 2.0),
        synchronize_joints=True,
    ),
    no_random_when_walking=True,
)
```

To visualize and compare all profiles:

```bash
python agile/rl_env/mdp/actions/velocity_profiles/test_profile_comparison.py
python agile/rl_env/mdp/actions/velocity_profiles/test_profile_comparison.py --save-figure
```

## Commands

Command generators produce the reference signals that the policy must track. Located in
`agile/rl_env/mdp/commands/`.

### Velocity Commands

**`UniformNullVelocityCommand`** / `UniformNullVelocityCommandCfg`
: Generates random velocity commands (linear x, y and angular yaw) with a configurable
  fraction of environments receiving zero-velocity ("stance") commands. Extends Isaac Lab's
  `UniformVelocityCommand` with:
  - **EMA smoothing**: Smooth velocity measurement for reward computation.
  - **Minimum velocity norm**: Commands below this threshold are zeroed out.
  - **Bias sampling**: Option to sample more low-speed commands for better stance training.
  - **Command filtering**: Per-axis low-pass filtering for smooth command transitions.

### Velocity + Height Commands

**`UniformVelocityBaseHeightCommand`** / `UniformVelocityBaseHeightCommandCfg`
: Extends velocity commands with a base height command. Includes:
  - A **minimum walk height**: Below this, velocity commands are scaled down to prevent
    walking while crouched.
  - **Squatting threshold**: Zeroes velocities when transitioning to low heights.
  - **Height sensor integration**: Uses a ray caster to measure height above terrain.

### Velocity + Height + Gait Commands

**`UniformVelocityGaitBaseHeightCommand`** / `UniformVelocityGaitBaseHeightCommandCfg`
: Adds gait phase information to velocity + height commands. Provides `gait_process`
  (current phase in [0, 1]) and `gait_frequency` signals used by gait-cycle rewards
  to enforce proper foot timing.

### Trajectory Tracking Commands

**`TrackingCommand`** / `TrackingCommandCfg`
: Generates reference poses from pre-recorded YAML trajectory files. Tracks:
  - Anchor body position and orientation (global reference)
  - Joint positions for specified tracked joints
  - Object position and orientation (if object tracking is configured)
  - Automatic peak detection for pick-and-place phase gating

## Observations

Observation terms define what the policy sees. Located in `agile/rl_env/mdp/observations/`.

### Observation Groups

Task configs define observation groups as nested `ObsGroup` dataclasses. Common groups:

- **`policy`**: Observations available to the deployed policy (proprioceptive only).
- **`critic`**: Additional observations for the critic during training (can include privileged info).
- **`teacher`**: Privileged observations for teacher policies (e.g., terrain height scans).

### Standard Observations

Most observations come from Isaac Lab's built-in terms:

| Term | Description |
|------|-------------|
| `base_ang_vel` | Base angular velocity in body frame |
| `projected_gravity` | Gravity vector projected into body frame (orientation indicator) |
| `joint_pos_rel` | Joint positions relative to defaults |
| `joint_vel_rel` | Joint velocities relative to defaults |
| `last_action` | Previous policy action |
| `generated_commands` | Current command vector |
| `height_scan` | Terrain height scan from ray caster (privileged) |

### Custom Observations

Defined in `observations/observations_io.py`:

| Term | Description |
|------|-------------|
| `velocity_height_command` | Velocity + height command vector for evaluation logging |
| `joint_acc` | Joint accelerations |

### Tracking Observations

Defined in `observations/tracking_observations.py` for the pick-and-place task. These
provide the current and target states for trajectory tracking.

### History Stacking

Observation groups support `history_length` to stack multiple timesteps. For example,
`history_length=5` concatenates the last 5 observation vectors, giving the policy
temporal context without recurrence.

## Terminations

Termination conditions end episodes early. Located in `agile/rl_env/mdp/terminations.py`.

### Standard Terminations

| Term | Description |
|------|-------------|
| `time_out` | Episode exceeds maximum length (marked as timeout, not failure). |
| `illegal_ground_contact` | Non-foot body contacts ground above force threshold while below minimum height. |
| `illegal_base_height` | Base height drops below threshold (adjusted for terrain). |

### Adaptive Terminations

| Class | Description |
|-------|-------------|
| `fall_from_max_height` | Terminates when the robot falls a configurable distance below its peak achieved height. More adaptive than fixed thresholds since it is relative to progress. Clamps maximum trackable height to ignore jumping. |
| `no_height_progress` | Terminates if no upward progress is made within a time window. More forgiving than `fall_from_max_height` -- does not punish falling after reaching standing height, only lack of any progress. |
| `standing` | Terminates when the robot stands above a height for a specified duration (used as a success condition in stand-up). |

### Trajectory Terminations

| Function | Description |
|----------|-------------|
| `bad_base_pose` | Base position error exceeds threshold from reference trajectory. |
| `bad_base_rotation` | Base orientation error exceeds threshold from reference. |
| `bad_joint_pos` | Joint position error exceeds threshold from reference. |
| `out_of_bound` | Object leaves a defined bounding box (supports reference frame transforms). |
| `link_distance` | Distance between two specified links is outside allowed range. |

## Events

Event terms handle environment resets and domain randomization. Located in
`agile/rl_env/mdp/events/`.

### Reset Events

| Term | Description |
|------|-------------|
| `reset_joints_around_default` | Reset joint positions and velocities to random values around defaults, clipped to soft limits. |
| `reset_root_state_uniform` | Reset robot base pose (Isaac Lab built-in). |

### Randomization Events

| Class | Description |
|-------|-------------|
| `disable_joints` | Temporarily disable specified joints for a random duration during an episode. Simulates actuator failures for robustness. |

### Fallen State Management

For the stand-up task, specialized event infrastructure manages pre-collected fallen states:

- **`FallenStateDataset`** (`events/fallen_state_dataset.py`): Manages collection and
  storage of fallen robot states. Spawns robots, lets them fall, and records the resulting
  joint positions and velocities.
- **`FallenStateCache`** (`events/fallen_state_cache.py`): Disk caching with automatic
  invalidation when terrain configuration changes.
- **`reset_from_fallen_dataset`** (`events/reset_from_fallen_dataset.py`): Episode reset
  event that samples from the fallen state dataset instead of simulating falls in real time.

## Curriculum

Curriculum terms adjust training difficulty over time. Located in
`agile/rl_env/mdp/curriculums/`.

### Terrain Curriculum (`task_curriculum.py`)

**`initial_pose_curriculum`**
: Progresses robots through terrain difficulty levels based on distance walked.
  Robots that walk far enough move to harder terrains; robots that underperform
  move to easier ones.

### Effort Limit Curriculum (`task_curriculum.py`)

**`effort_limit_curriculum`**
: Gradually decreases actuator effort limits over training. Starts with inflated
  limits for easier initial exploration, then decays geometrically toward the real
  hardware limits. Adjusts both effort limits and saturation effort for DC motors.

**`effort_limit_curriculum_traveled_distance`**
: Same concept, but triggered by traveled distance rather than another curriculum's state.

### Harness/Lift Curriculum (`task_curriculum.py`)

Various curriculum terms reduce assistance forces over training:

- **`remove_harness`**: Gradually reduces harness stiffness and damping to zero.
- **`adaptive_lift_curriculum`**: Reduces lift force based on standing success rate.

### Randomization Curriculum (`randomization_curriculum.py`)

**`increase_event_randomization`**
: Increases the range of domain randomization parameters over training. Scales event
  parameter ranges from an initial fraction to a terminal fraction, based on another
  curriculum's progress.

## Terrains

Terrain configurations for rough terrain training. Located in `agile/rl_env/mdp/terrains/`.

### Pre-configured Terrain Sets

Defined in `terrains.py`:

| Config | Description |
|--------|-------------|
| `ROUGH_TERRAIN_CFG` | Full difficulty range with random grid boxes, random rough surfaces, and slope ramps. |
| `LESS_ROUGH_TERRAIN_CFG` | Reduced difficulty variant for initial training stages. |

### Custom Terrain Types

Defined in `hf_terrains.py` / `hf_terrains_cfg.py`:

**`HfRandomUniformTerrainDifficultyCfg`**
: Height-field based random uniform terrain where noise range scales with difficulty level.
  Provides smooth progression from flat to highly irregular surfaces.

## Actuators

Custom actuator models that simulate real hardware behavior. Located in
`agile/rl_env/mdp/actuators/`.

### Delayed DC Motor (`DelayedDCMotor` / `DelayedDCMotorCfg`)

Extends Isaac Lab's `DCMotor` with communication delay simulation:

- **Random delay**: Each environment gets a random delay between `min_delay` and `max_delay`
  timesteps, sampled at reset.
- **Delay buffers**: Separate buffers for position, velocity, and effort signals.
- **Torque-speed curve**: Models the DC motor characteristic where available torque
  decreases with joint velocity.

### Delayed Implicit Actuator (`DelayedImplicitActuator` / `DelayedImplicitActuatorCfg`)

Extends Isaac Lab's `ImplicitActuator` with the same delay mechanism. Used for
locomotion-height tasks where the actuator model is simpler but delay is still important
for sim-to-real transfer.

```{tip}
Actuator delay is critical for sim-to-real transfer. Real robots have non-negligible
communication latency between the policy computer and joint controllers. Training with
randomized delays makes the policy robust to this variation.
```

## Symmetry

Morphological symmetry augmentation for data-efficient training. Located in
`agile/rl_env/mdp/symmetry/`.

### Purpose

Bipedal robots have left-right symmetry: a mirrored observation should produce a mirrored
action. AGILE leverages this by augmenting the training data with symmetry-transformed
samples, effectively doubling the data efficiency.

### Robot-Specific Implementations

**`lr_mirror_G1`** (`symmetry_g1.py`)
: Left-right mirror function for the Unitree G1 robot. Transforms observations and
  actions by swapping left/right joint indices and negating lateral quantities.

**`lr_mirror_T1`** (`symmetry_t1.py`)
: Left-right mirror function for the Booster T1 robot.

### Observation Mirror Primitives (`observations.py`)

Building blocks used by robot-specific mirror functions:

| Function | Description |
|----------|-------------|
| `lr_mirror_base_lin_vel` | Negate y-component of base linear velocity. |
| `lr_mirror_base_ang_vel` | Negate x and z components of base angular velocity. |
| `lr_mirror_projected_gravity` | Negate y-component of projected gravity. |
| `mirror_velocity_commands` | Negate y and yaw velocity commands. |
| `mirror_height_scan_left_right` | Swap left and right height scan regions. |
| `mirror_gait_cycle_commands` | Swap gait phase for left and right legs. |

## Stability Terms (`stability_terms.py`)

Utility functions used by multiple MDP components for computing stability-related
quantities (e.g., center of mass, support polygon membership).

## Utility Functions (`utils.py`)

Shared helper functions used across MDP components:

| Function | Description |
|----------|-------------|
| `get_robot_cfg` | Get robot asset with proper configuration, creating defaults if needed. |
| `get_contact_sensor_cfg` | Get contact sensor with proper body ID resolution. |
| `get_body_velocities_and_forces` | Extract body velocities and contact forces for specified bodies. |
| `transform_to_asset_frame` | Transform world positions to asset-local coordinates. |
| `transform_to_body_frame` | Transform world positions to a body's local frame. |
| `compute_asset_aabb` | Compute axis-aligned bounding box for an asset. |
| `get_joint_indices` | Get joint indices for a specified body part (lower, upper, or whole body). |
