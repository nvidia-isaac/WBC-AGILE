# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration classes for velocity profiles."""

from __future__ import annotations

from typing import Literal

# Conditional import for Isaac Lab compatibility
try:
    from isaaclab.utils import configclass
except ImportError:
    # Fallback to standard dataclass when Isaac Lab is not available
    from dataclasses import dataclass as configclass


@configclass
class VelocityProfileBaseCfg:
    """Base configuration for all velocity profiles."""

    position_tolerance: float = 0.001
    """Tolerance for determining if trajectory is complete (in radians)."""

    velocity_tolerance: float = 0.01
    """Tolerance for determining if velocity is effectively zero (rad/s)."""

    enable_position_limits: bool = True
    """Whether to enforce joint position limits during trajectory."""

    enable_velocity_limits: bool = True
    """Whether to enforce joint velocity limits during trajectory."""


@configclass
class EMAVelocityProfileCfg(VelocityProfileBaseCfg):
    """Configuration for Exponential Moving Average (EMA) velocity profile."""

    ema_coefficient_range: tuple[float, float] = (0.01, 0.02)
    """Range for EMA coefficient (α). Higher values = faster convergence to target.

    The position update follows: pos_new = α * target + (1 - α) * pos_current
    where α is randomly sampled from this range for each trajectory.
    """

    use_adaptive_scale: bool = False
    """If True, coefficient adapts based on distance to target."""

    synchronize_joints: bool = True
    """If True, all joints use the same EMA coefficient for synchronized motion."""


@configclass
class LinearVelocityProfileCfg(VelocityProfileBaseCfg):
    """Configuration for true linear velocity profile with constant velocity."""

    velocity_range: tuple[float, float] = (0.5, 2.0)
    """Range for constant velocity in rad/s. Randomized per trajectory."""

    synchronize_joints: bool = True
    """If True, all joints complete trajectory simultaneously."""

    acceleration_time: float = 0.0
    """Time to accelerate to target velocity (0 for instantaneous)."""


@configclass
class TrapezoidalVelocityProfileCfg(VelocityProfileBaseCfg):
    """Configuration for trapezoidal velocity profile."""

    acceleration_range: tuple[float, float] = (0.5, 2.0)
    """Range for acceleration in rad/s^2. Randomized per joint."""

    max_velocity_range: tuple[float, float] = (0.5, 2.0)
    """Range for maximum velocity in rad/s. Randomized per joint."""

    deceleration_range: tuple[float, float] | None = None
    """Range for deceleration in rad/s^2. If None, uses acceleration_range."""

    min_cruise_ratio: float = 0.1
    """Minimum ratio of trajectory that should be at cruise velocity.
    Prevents degenerate triangular profiles. Range [0, 1]."""

    synchronize_joints: bool = True
    """If True, all joints complete trajectory simultaneously."""

    time_scaling_method: Literal["max_time", "average_time"] = "max_time"
    """Method for synchronizing joint trajectories:
    - 'max_time': All joints use the slowest joint's time
    - 'average_time': All joints use average completion time
    """

    use_smooth_start: bool = True
    """If True, inherit current velocity when starting new trajectory for smoother motion."""
