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


import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

from .height_command import SmoothHeightCommand


@configclass
class SmoothHeightCommandCfg(CommandTermCfg):
    """Configuration for the smooth height command term.

    Commands a target height that ramps smoothly between randomly sampled
    targets at random velocities. Never changes instantly.
    """

    class_type: type = SmoothHeightCommand

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    @configclass
    class Ranges:
        """Ranges for height sampling."""

        height: tuple[float, float] = (0.0, 0.75)
        """Range of target heights to sample (meters)."""

    ranges: Ranges = Ranges()
    """Ranges for command sampling."""

    @configclass
    class OffsetCfg:
        """The offset of the tracked point from the body frame origin."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the body's local frame. Defaults to (0.0, 0.0, 0.0)."""

    body_name: str = "torso_link"
    """Body whose height is tracked and commanded."""

    offset: OffsetCfg = OffsetCfg()
    """Offset of the tracked point from the body frame origin. Defaults to zero offset."""

    height_sensor: str = "height_measurement_sensor"
    """Name of the raycaster sensor for terrain-relative height measurement."""

    velocity_range: tuple[float, float] = (0.05, 0.5)
    """Range of ramp velocities (m/s). Min > 0 ensures no instant transitions.
    Use very large values (e.g. 1000.0) for instant jumps."""

    settle_time_s: float = 1.0
    """Seconds after each resample before tracking is expected.

    During the settle period, the ``settled`` property returns False and
    the height-error metric ignores these steps.  Also used as the grace
    period for the episode-mean height-error metric.
    """

    standing_ratio: float = 0.0
    """Fraction of resamples that set the target to the maximum height (standing). Defaults to 0.0."""

    flat_ratio: float = 0.0
    """Fraction of resamples that set the target to the minimum height (lying down). Defaults to 0.0."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.standing_ratio + self.flat_ratio > 1.0:
            raise ValueError(
                f"standing_ratio ({self.standing_ratio}) + flat_ratio ({self.flat_ratio}) must not exceed 1.0"
            )

    goal_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/height_goal",
        markers={
            "cube": sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity=0.7),
            ),
        },
    )
    """Configuration for the goal height visualization marker (green)."""

    measured_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/height_measured",
        markers={
            "cube": sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0), opacity=0.7),
            ),
        },
    )
    """Configuration for the measured height visualization marker (blue)."""
