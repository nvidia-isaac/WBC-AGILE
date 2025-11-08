# Velocity Profiles

This module implements different velocity profiles for trajectory generation in random action terms.

## Available Profiles

### 1. EMAVelocityProfile (Exponential Moving Average)
- **Behavior**: Exponential approach to target position
- **Formula**: `pos = α * target + (1 - α) * current`
- **Use case**: Smooth, natural-looking motion with soft convergence
- **Parameter**: `ema_coefficient_range` - Controls convergence speed

### 2. LinearVelocityProfile (Constant Velocity)
- **Behavior**: Constant velocity motion
- **Formula**: `pos = initial + velocity * time`
- **Use case**: Predictable, time-based control
- **Parameter**: `velocity_range` - Constant velocity in rad/s

### 3. TrapezoidalVelocityProfile
- **Behavior**: Three-phase motion (acceleration, cruise, deceleration)
- **Use case**: Physically realistic motion with controlled acceleration
- **Parameters**: `acceleration_range`, `max_velocity_range`, `deceleration_range`

## Testing

Run the comparison test to visualize all three profiles:

```bash
# View plots without saving
python agile/rl_env/mdp/actions/velocity_profiles/test_profile_comparison.py

# Save plots to file
python agile/rl_env/mdp/actions/velocity_profiles/test_profile_comparison.py -s
# or
python agile/rl_env/mdp/actions/velocity_profiles/test_profile_comparison.py --save-figure
```

The test:
- Runs standalone without Isaac Sim dependencies
- Compares all three profiles with 2 joints
- Generates plots showing position, velocity, and acceleration trajectories
- Demonstrates synchronized joint motion

## Usage Example

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

## Implementation Notes

- All profiles use fully vectorized batch operations for efficiency
- Joints can be synchronized to complete trajectories simultaneously
- Position and velocity limits are enforced
- Profile parameters are randomized per trajectory
