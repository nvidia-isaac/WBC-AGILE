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


from dataclasses import MISSING, field
from typing import Literal

from isaaclab.envs import mdp
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg  # noqa: F401
from isaaclab.utils import configclass  # noqa: F401

# pyright: reportGeneralTypeIssues=false
from agile.rl_env.mdp.actions.harness_action import HarnessAction
from agile.rl_env.mdp.actions.lift_action import LiftAction

# Import GUI action class for registration
from .joint_pos_gui_action import JointPositionGUIAction  # noqa: F401, E402
from .random_actions import RandomPositionAction  # noqa: F401, E402

# Import velocity profile configurations
from .velocity_profiles import (
    EMAVelocityProfileCfg,
    VelocityProfileBaseCfg,
)  # noqa: F401, E402

##
# Joint actions.
##


@configclass
class RandomActionCfg(mdp.JointActionCfg):
    """Configuration for the base random action term.

    See :class:`RandomAction` for more details.
    """

    class_type: type[ActionTerm] = RandomPositionAction

    actuator_group: str | None = None
    """The actuator group to apply the action to"""

    joint_names: list[str] = []
    """List of joint names or regex expressions that the action will be mapped to.
    If empty, all joints will be included."""

    joint_names_exclude: list[str] = []
    """List of joint names or regex expressions that the action will be excluded from.
    If empty, no joints will be excluded. Only one of :attr:`actuator_group`, :attr:`joint_names` or
    :attr:`joint_names_exclude` can be provided."""

    sample_range: tuple[float, float] = (0.0, 1.0)
    """Range of the resampling time. Defaults to (0.0, 1.0) in seconds."""

    velocity_profile_cfg: VelocityProfileBaseCfg = field(default_factory=lambda: EMAVelocityProfileCfg())
    """Velocity profile configuration. Defaults to EMA profile for backward compatibility."""

    command_name: str = "base_velocity"
    """Name of the command to use for the action term. Defaults to "base_velocity"."""

    no_random_when_walking: bool = False
    """If True, no randomization will be applied when the robot is walking. Defaults to False."""

    joint_pos_limits: dict[str, tuple[float, float]] | None = None
    """Dictionary mapping joint names to custom position limits (lower_limit, upper_limit).
    If provided, overrides the default joint limits from the robot asset. Defaults to None."""


@configclass
class JointPositionGUIActionCfg(mdp.JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    # Register the GUI action so that the action manager instantiates our implementation.
    class_type: type[ActionTerm] = JointPositionGUIAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """
    max_stiffness: float = 200.0
    """Maximum stiffness for the P-gain slider. Defaults to 200.0."""
    max_damping: float = 25.0
    """Maximum damping for the D-gain slider. Defaults to 25.0."""

    mirror_actions: bool = True
    """Whether to mirror the actions. Defaults to True."""

    robot_type: Literal["t1", "g1"] = "t1"
    """The type of robot to use. Defaults to 't1'."""


@configclass
class HarnessActionCfg(ActionTermCfg):
    """Action term to simulate a simplified harness.

    Applies external forces and torques to the root body to prevent it from falling.
    """

    class_type: type[ActionTerm] = HarnessAction
    """The type of the action term."""
    root_name: str = MISSING
    """The name of the root joint of the articulation."""
    stiffness_torques: float = 0.0
    """The stiffness of the harness. Defaults to 0.0."""
    damping_torques: float = 0.0
    """The damping of the harness. Defaults to 0.0."""
    stiffness_forces: float = 0.0
    """The stiffness of the forces applied by the harness. Defaults to 0.0."""
    damping_forces: float = 0.0
    """The damping of the forces applied by the harness. Defaults to 0.0."""
    force_limit: float = 0.0
    """The force limit of the harness. Defaults to 0.0."""
    torque_limit: float = 0.0
    """The torque limit of the harness. Defaults to 0.0."""
    target_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w, x, y, z
    """The target orientation of the harness in quaternion format. Defaults to (1.0, 0.0, 0.0, 0.0)."""
    height_sensor: str = "height_measurement_sensor"
    """The name of the height sensor to use for the harness action. Defaults to "height measurement_sensor"."""
    target_height: float = 0.72
    """The target height of the harness action. Defaults to 0.72."""
    command_name: str | None = None
    """Name of the command term for height commands (e.g., 'base_velocity'). If None, uses fixed target_height."""


@configclass
class LiftActionCfg(ActionTermCfg):
    """Action term to simulate a lift.

    Applies external forces to lift the robot up.
    """

    class_type: type[ActionTerm] = LiftAction
    """The type of the action term."""
    link_to_lift: str = MISSING
    """The name of the root joint of the articulation."""
    stiffness_forces: float = 0.0
    """The stiffness of the forces applied by the harness. Defaults to 0.0."""
    damping_forces: float = 0.0
    """The damping of the forces applied by the harness. Defaults to 0.0."""
    force_limit: float = 0.0
    """The force limit of the harness. Defaults to 0.0."""
    height_sensor: str = "height_measurement_sensor"
    """The name of the height sensor to use for the harness action. Defaults to "height measurement_sensor"."""
    target_height: float = 0.71
    """The target height of the harness action. Defaults to 0.71."""
    height_command: str | None = None
    """If a heights are commanded, the command term can be added here"""
    start_lifting_time_s: float = 0.0
    """After how many seconds the lift should start"""
    lifting_duration_s: float = 10.0
    """How many seconds the lift should take to move from start to end"""
