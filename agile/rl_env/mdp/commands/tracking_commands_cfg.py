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

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from agile.rl_env.mdp.commands.tracking_commands import TrackingCommand


@configclass
class TrackingCommandCfg(CommandTermCfg):
    """Configuration for the tracking command term."""

    class_type: type = TrackingCommand
    resampling_time_range: tuple[float, float] = (
        1e6,
        1e6,
    )  # no resampling based on time

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    anchor_body_name: str | None = None
    """Name of the body to use as the anchor for tracking. If None, uses the root body."""

    object_name: str | None = None
    """Name of the object in the environment to track. If None, object tracking is disabled."""

    joint_names: list[str] = [""]
    """The names of the joints to track. All other joints will be reset to default positions."""

    make_quat_unique: bool = True
    """Whether to make the quaternion unique or not.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    file_path: str = ""
    """Path to the YAML file containing the motion data (joint positions in Isaac order)."""

    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Offset of the root pose in the environment frame."""

    update_goal_on_reach: bool = False
    """Whether to update the goal when the goal is reached."""

    goal_reach_threshold: float = 0.1
    """Threshold for considering the goal as reached."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_marker"
    )
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
