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

"""Configuration for the Object Pose GUI Action."""

from __future__ import annotations

import math
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    pass


@configclass
class ObjectPoseGUIActionCfg(ActionTermCfg):
    """Configuration for the object pose GUI action term.

    This action term allows interactive control of a rigid object's 6DoF pose via a
    DearPyGui window. The user can adjust position (x, y, z) and orientation
    (roll, pitch, yaw) using sliders.

    Attributes:
        object_name: Name of the rigid object in the scene to control.
        position_limits: Dictionary mapping axis names to (min, max) tuples in meters.
        rotation_limits: Dictionary mapping axis names to (min, max) tuples in radians.
        enable_velocity_control: Whether to show velocity control sliders.
        disable_gravity: Whether to disable gravity on the object during GUI control.
        gui_window_title: Title of the DearPyGui window.
    """

    class_type: type[ActionTerm] = field(default_factory=lambda: _get_object_pose_gui_action_class())
    """The class type for this action term."""

    # Object to control (uses asset_name from base ActionTermCfg)
    asset_name: str = "object"
    """Name of the rigid object entity in the scene. Defaults to 'object'."""

    # Position limits (meters)
    position_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-2.0, 2.0),
            "y": (-2.0, 2.0),
            "z": (0.0, 2.0),
        }
    )
    """Position limits for each axis in meters. Defaults to ±2m for x/y, 0-2m for z."""

    # Rotation limits (radians)
    rotation_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "roll": (-math.pi, math.pi),
            "pitch": (-math.pi, math.pi),
            "yaw": (-math.pi, math.pi),
        }
    )
    """Rotation limits for each Euler angle in radians. Defaults to ±π for all axes."""

    # GUI settings
    enable_velocity_control: bool = False
    """Whether to enable velocity control sliders. Defaults to False."""

    disable_gravity: bool = True
    """Whether to disable gravity on the object during GUI control. Defaults to True.
    This makes it easier to position objects without them falling."""

    gui_window_title: str = "Object Pose Controller"
    """Title of the DearPyGui window. Defaults to 'Object Pose Controller'."""

    gui_window_width: int = 500
    """Width of the GUI window in pixels. Defaults to 500."""

    gui_window_height: int = 400
    """Height of the GUI window in pixels. Defaults to 400."""


def _get_object_pose_gui_action_class() -> type[ActionTerm]:
    """Lazy import to avoid circular dependency."""
    from .object_pose_gui_action import ObjectPoseGUIAction

    return ObjectPoseGUIAction
