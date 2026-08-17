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


from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass

from agile.rl_env.mdp.commands.commands import (
    UniformNullVelocityCommand,
    UniformVelocityBaseHeightCommand,
    UniformVelocityGaitBaseHeightCommand,
)


@configclass
class UniformNullVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformNullVelocityCommand

    ema_smoothing_param: float = 1.0
    r"""Parameters for the exponential moving average smoothing of the command. It is used to smooth the
    velocity of the robot:

    .. math::

        \text{vel_robot_smoothed} = \text{ema_smoothing_params} \times \text{vel_robot} + (1 - \text{ema_smoothing_params}) \times \text{vel_command}

    where :math:`\text{vel_robot}` is the velocity of the robot, :math:`\text{vel_command}` is the command velocity,
    and :math:`\text{ema_smoothing_params}` is the smoothing parameter.

    1.0 means no smoothing.
    """  # noqa: E501, W605

    min_vel_norm: float = 0.1
    """Minimum velocity norm,velocity commands with a norm less than this value are set to 0"""

    bias_sampling: bool = False
    """Enable biased sampling toward lower speeds. When True, samples velocities from a piecewise
    uniform distribution that favors low speeds."""

    bias_sampling_probability: float = 0.6
    """Probability of sampling from the low-speed range when bias_sampling is True.
    For example, 0.5 means 50% of samples come from the low-speed range."""

    bias_sampling_speed: float = 0.2
    """Maximum speed for the low-speed range when bias_sampling is True (in m/s for linear velocities,
    rad/s for angular velocities). Commands are sampled from [-bias_sampling_speed, bias_sampling_speed]
    with probability bias_sampling_probability."""

    command_filter_alpha_ranges: dict[str, tuple[float, float] | None] | None = None
    """Per-axis low-pass filter coefficient ranges for smooth command transitions.

    Keys are axis names that correspond to command dimensions.
    Base class supports: 'lin_vel_x', 'lin_vel_y', 'ang_vel_z'
    Subclasses can add additional axes (e.g., 'base_height' for height commands).

    Values are tuples (min_alpha, max_alpha) for random sampling per environment.
    Alpha values should be in [0, 1] where:
    - 1.0 = no filtering (instant response)
    - 0.0 = maximum filtering (no change)
    - Typical values: 0.05-0.3 for smooth transitions

    Example:
        command_filter_alpha_ranges = {
            "lin_vel_x": (0.05, 0.2),  # Smooth linear x velocity
            "lin_vel_y": (0.05, 0.2),  # Smooth linear y velocity
            "ang_vel_z": (0.1, 0.3),   # Slightly faster angular response
            # "base_height": (0.02, 0.1),  # Height filtering (if using height commands)
        }

    If not provided or empty dict, no filtering is applied (backward compatible).
    Subclasses handle their additional axes independently.
    """


@configclass
class UniformVelocityBaseHeightCommandCfg(UniformNullVelocityCommandCfg):
    """Configuration for velocity and base height command generator."""

    class_type: type = UniformVelocityBaseHeightCommand

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity and base height commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """

        base_height: tuple[float, float] = MISSING
        """Range for the base height command (in m)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity and base height commands."""

    min_walk_height: float = 0.4
    """Minimum height for the robot to walk.
    If the commanded height is below this value, velocity commands are scaled down from 1 to 0.
    """

    random_height_during_walking: bool = False
    """If false, height command is only given for standing envs"""

    squatting_threshold: float = 0.7
    """Height threshold below which the robot is considered to be squatting.
    When transitioning from standing to a height below this threshold, velocities are zeroed."""

    default_height: float = MISSING
    """The default waling height"""

    height_sensor: str = MISSING
    """The height sensor to measure the base height"""

    root_name: str = MISSING
    """The name of the body to track the """

    bias_height_randomization: bool = False
    """If true, bias the height randomization towards lower heights."""

    lower_height_bias: float = 0.8
    """The bias for the lower height range."""

    sample_middle_height: float = 0.5
    """The middle height to sample from."""


@configclass
class UniformVelocityGaitBaseHeightCommandCfg(UniformVelocityBaseHeightCommandCfg):
    """Configuration for velocity and base height command generator with gait phase."""

    class_type: type = UniformVelocityGaitBaseHeightCommand

    gait_frequency_range: tuple[float, float] = (1.0, 2.0)
    """The range of the gait frequency in Hz."""
