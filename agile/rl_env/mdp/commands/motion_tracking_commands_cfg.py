# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from agile.rl_env.mdp.commands.motion_tracking_commands import MotionCommand


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING  # type: ignore[assignment]

    motion_file: str = MISSING  # type: ignore[assignment]
    anchor_body_name: str = MISSING  # type: ignore[assignment]
    body_names: list[str] = MISSING  # type: ignore[assignment]

    motion_joint_names: list[str] | None = None
    """Joint names in the motion file's joint ordering.

    If provided, the loader will remap ``joint_pos`` / ``joint_vel`` from this
    ordering to the robot's joint ordering at load time.  When ``None``
    (default) no remapping is performed and the motion file is assumed to
    already match the robot's joint order."""

    motion_body_names: list[str] | None = None
    """Full list of body names in the motion file's body ordering.

    If provided, the loader will use indices derived from this list (instead
    of the robot's body indices) when reading ``body_pos_w``, ``body_quat_w``,
    etc. from the motion file.  This is needed when the motion data was
    recorded from a different robot model (e.g. URDF) whose body ordering
    differs from the simulation robot (e.g. USD).  When ``None`` (default)
    the motion file is assumed to share the robot's body ordering."""

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
