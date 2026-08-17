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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from .height_command_cfg import SmoothHeightCommandCfg


class SmoothHeightCommand(CommandTerm):
    """Command term that generates smooth height targets for stand-up tasks.

    Samples random target heights and ramp velocities at each resample interval.
    Between resamples, the command smoothly ramps toward the target at the sampled
    velocity (linear interpolation, never instant).

    On reset, initializes to the robot's current body height above ground.
    """

    cfg: SmoothHeightCommandCfg

    def __init__(self, cfg: SmoothHeightCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._height_sensor: RayCaster = env.scene.sensors[cfg.height_sensor]

        # Resolve body index
        body_ids, _ = self.robot.find_bodies(cfg.body_name)
        if len(body_ids) == 0:
            raise ValueError(f"Body '{cfg.body_name}' not found in robot.")
        self._body_id = body_ids[0]

        # Local frame offset for the tracked point.
        # This allows tracking a point that is not at the body frame origin — e.g.
        # the CoM or a sensor location.  The offset is expressed in the body's local
        # frame and gets rotated into world frame each step via quat_apply.
        self._offset_local = torch.tensor(cfg.offset.pos, device=self.device, dtype=torch.float32)

        # Buffers
        self._target_height = torch.zeros(self.num_envs, device=self.device)
        self._current_height_cmd = torch.zeros(self.num_envs, device=self.device)
        self._velocity = torch.zeros(self.num_envs, device=self.device)

        # Metrics — episode mean height error (with resample grace period)
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

        # Episode error accumulation buffers
        self._episode_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_step_count = torch.zeros(self.num_envs, device=self.device)

        # Steps since last resample (for settle / grace period)
        self._steps_since_resample = torch.zeros(self.num_envs, device=self.device)
        self._settle_steps = int(cfg.settle_time_s / env.step_dt)

    @property
    def command(self) -> torch.Tensor:
        """The current smooth height command. Shape: [num_envs, 1]."""
        return self._current_height_cmd.unsqueeze(-1)

    @property
    def target_height(self) -> torch.Tensor:
        """The current smooth height target (for lift action / rewards). Shape: [num_envs]."""
        return self._current_height_cmd

    @property
    def measured_height(self) -> torch.Tensor:
        """The actual body height above ground. Shape: [num_envs]."""
        return self._measure_height()

    @property
    def settled(self) -> torch.Tensor:
        """Bool tensor [num_envs] — True when enough time has passed since last resample."""
        return self._steps_since_resample > self._settle_steps

    @property
    def relaxation_intensity(self) -> torch.Tensor:
        """Float tensor [num_envs] in [0, 1]. 0 when command >= 0, 1 at the most negative command."""
        min_height = self.cfg.ranges.height[0]
        if min_height >= 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.clamp(-self._current_height_cmd / (-min_height), 0.0, 1.0)

    def _tracked_point_pos_w(self) -> torch.Tensor:
        """Compute the world position of the tracked point (body origin + rotated offset). Shape: [num_envs, 3]."""
        body_pos_w = self.robot.data.body_pos_w.torch[:, self._body_id, :]  # (num_envs, 3)
        body_quat_w = self.robot.data.body_quat_w.torch[:, self._body_id, :]  # (num_envs, 4)
        offset_w = quat_apply(body_quat_w, self._offset_local.expand(self.num_envs, -1))
        return body_pos_w + offset_w

    def _measure_height(self) -> torch.Tensor:
        """Measure the tracked point's height above ground using the height sensor."""
        point_z_w = self._tracked_point_pos_w()[:, 2]
        ground_height = torch.mean(self._height_sensor.data.ray_hits_w[..., 2], dim=-1)
        return point_z_w - ground_height

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Sample new target height and ramp velocity."""
        self._steps_since_resample[env_ids] = 0

        n = len(env_ids)
        r = self.cfg.ranges

        # Sample target heights with biased distribution
        roll = torch.rand(n, device=self.device)
        standing_mask = roll < self.cfg.standing_ratio
        flat_mask = (~standing_mask) & (roll < self.cfg.standing_ratio + self.cfg.flat_ratio)
        uniform_mask = ~standing_mask & ~flat_mask

        heights = torch.empty(n, device=self.device)
        heights[standing_mask] = r.height[1]
        heights[flat_mask] = r.height[0]
        heights[uniform_mask] = torch.empty(uniform_mask.sum().item(), device=self.device).uniform_(
            r.height[0], r.height[1]
        )
        self._target_height[env_ids] = heights

        # Sample ramp velocities
        v_min, v_max = self.cfg.velocity_range
        self._velocity[env_ids] = torch.empty(n, device=self.device).uniform_(v_min, v_max)

    def _update_command(self) -> None:
        """Ramp current command toward target at sampled velocity."""
        dt = self._env.step_dt
        diff = self._target_height - self._current_height_cmd
        max_step = self._velocity * dt

        # Move toward target, clamping step size to avoid overshoot
        step = torch.clamp(diff, -max_step, max_step)
        self._current_height_cmd += step

    def _update_metrics(self) -> None:
        """Track episode-mean height error, skipping settle period and negative commands."""
        error = torch.abs(self.measured_height - self._current_height_cmd)

        # Accumulate only when settled AND command is non-negative
        self._steps_since_resample += 1
        valid = (self._steps_since_resample > self._settle_steps) & (self._current_height_cmd >= 0)
        self._episode_error_sum += error * valid
        self._episode_step_count += valid.float()
        self.metrics["height_error"] = self._episode_error_sum / self._episode_step_count.clamp(min=1)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset command to current body height for given envs."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset episode error accumulators
        self._episode_error_sum[env_ids] = 0.0
        self._episode_step_count[env_ids] = 0.0

        # Initialize command to current measured height, clamped to configured range
        r = self.cfg.ranges
        self._current_height_cmd[env_ids] = self._measure_height()[env_ids].clamp(r.height[0], r.height[1])

        # Let parent handle resampling (calls _resample_command)
        result: dict[str, float] = super().reset(env_ids)
        return result

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "_goal_marker"):
                self._goal_marker = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
                self._measured_marker = VisualizationMarkers(self.cfg.measured_visualizer_cfg)
            self._goal_marker.set_visibility(True)
            self._measured_marker.set_visibility(True)
        else:
            if hasattr(self, "_goal_marker"):
                self._goal_marker.set_visibility(False)
                self._measured_marker.set_visibility(False)

    def _debug_vis_callback(self, event: object) -> None:  # noqa ARG002
        if not self.robot.is_initialized:
            return
        tracked_pos = self._tracked_point_pos_w()
        ground_height = torch.mean(self._height_sensor.data.ray_hits_w[..., 2], dim=-1)

        # goal marker: commanded height, shifted in world X so it's visible
        goal_pos = tracked_pos.clone()
        goal_pos[:, 0] += 0.25
        goal_pos[:, 2] = ground_height + self._current_height_cmd
        self._goal_marker.visualize(goal_pos)

        # measured marker: actual tracked point height, same world X shift
        measured_pos = tracked_pos.clone()
        measured_pos[:, 0] += 0.25
        self._measured_marker.visualize(measured_pos)
