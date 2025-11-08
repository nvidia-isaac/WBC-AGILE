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

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.sensors import RayCaster
from isaaclab.utils import math as math_utils

# Only import the command class during type checking
if TYPE_CHECKING:
    from .commands_cfg import (
        UniformNullVelocityCommandCfg,
        UniformVelocityBaseHeightCommandCfg,
        UniformVelocityGaitBaseHeightCommandCfg,
    )


class UniformNullVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with min velocity command and traveled distance metric."""

    def __init__(self, cfg: UniformNullVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.metrics_cumulative: dict[str, torch.Tensor] = {}
        for k, v in self.metrics.items():  # type: ignore
            if "xy" in k:
                # store vectors not norms
                self.metrics_cumulative[k] = torch.zeros(env.num_envs, 2, device=self.device)
            else:
                self.metrics_cumulative[k] = torch.zeros_like(v)

        # for performance estimate
        self.metrics["traveled_distance"] = torch.zeros(env.num_envs, device=self.device)  # type: ignore
        self.traveled_distance = torch.zeros(env.num_envs, device=self.device)
        self.start_positions = self.robot.data.root_pos_w[:, :2].clone()

        self.min_vel_norm = cfg.min_vel_norm

        # smoothed velocity estimate
        self.smoothing_param = cfg.ema_smoothing_param
        self.vel_xy_smoothed = torch.zeros_like(self.robot.data.root_lin_vel_b[:, :2])
        self.angvel_smoothed = torch.zeros_like(self.robot.data.root_ang_vel_b[:, 2])

    def _update_metrics(self) -> None:
        # update smoothed velocity estimate

        vel_xy = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_quat_w),
            self.robot.data.root_lin_vel_w[:, :3],
        )[:, :2]
        self.vel_xy_smoothed = self.smoothing_param * vel_xy + (1 - self.smoothing_param) * self.vel_xy_smoothed
        self.angvel_smoothed = (
            self.smoothing_param * self.robot.data.root_ang_vel_w[:, 2]
            + (1 - self.smoothing_param) * self.angvel_smoothed
        )

        # logs data
        self.metrics_cumulative["error_vel_xy"] += self.vel_command_b[:, :2] - self.vel_xy_smoothed
        self.metrics_cumulative["error_vel_yaw"] += torch.abs(self.vel_command_b[:, 2] - self.angvel_smoothed)

        current_positions = self.robot.data.root_pos_w[:, :2]
        traveled_dist = torch.norm(current_positions - self.start_positions, dim=1)

        # norm of vector sum, not sum of norms
        normalizer = torch.clamp(self._env.episode_length_buf, min=1.0)

        self.metrics = {
            k: (v / normalizer if "xy" not in k else torch.norm(v, dim=-1) / normalizer)
            for k, v in self.metrics_cumulative.items()
        }
        self.metrics["traveled_distance"] = self.traveled_distance + traveled_dist

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = {k: v[env_ids].mean().item() for k, v in self.metrics.items()}

        self.traveled_distance[env_ids] = 0.0
        self.start_positions[env_ids] = self.robot.data.root_pos_w[env_ids, :2].clone()

        super().reset(env_ids)
        for _, v in self.metrics_cumulative.items():
            v[env_ids] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        current_positions = self.robot.data.root_pos_w[env_ids, :2]
        self.traveled_distance[env_ids] += torch.norm(current_positions - self.start_positions[env_ids], dim=1)
        self.start_positions[env_ids] = current_positions.clone()

        # set small velocity samples to zer
        too_small = self.vel_command_b.norm(dim=1) < self.min_vel_norm
        self.vel_command_b[too_small] = 0


class UniformVelocityBaseHeightCommand(UniformNullVelocityCommand):
    """Uniform velocity command generator with height command."""

    cfg: UniformVelocityBaseHeightCommandCfg
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: UniformVelocityBaseHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.target_height = torch.zeros(env.num_envs, device=self.device)
        self.base_height = torch.zeros(env.num_envs, device=self.device)

        self.error_height_log = 0.0
        self.random_height_during_walking = cfg.random_height_during_walking
        self.normal_walking_height = cfg.default_height

        # Track previous standing state for height randomization while standing.
        self.prev_stand_normal_height = torch.ones(env.num_envs, dtype=torch.bool, device=self.device)
        self.prev_stand_squat_height = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.prev_walk = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

        self.current_stand_normal_height = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.current_stand_squat_height = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        self.current_walk = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)

        # Raycaster to measure base height
        self._height_sensor: RayCaster = env.scene.sensors[cfg.height_sensor]  # type: ignore
        self._root_id, _ = self.robot.find_bodies(cfg.root_name)

    def __str__(self) -> str:
        msg = "UniformVelocityBaseHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        msg += f"\tBase height: {self.cfg.ranges.base_height}"
        msg += f"\tRoot name: {self.cfg.root_name}"
        msg += f"\tDefault hight: {self.cfg.default_height}"
        return msg

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample command with strict transition rules to prevent direct walking<->squatting."""
        super()._resample_command(env_ids)

        env_ids_tensor = torch.as_tensor(env_ids, device=self.device)
        # Sample the height command for all standing envs.
        # Start everyone with default height
        self.target_height[env_ids] = self.normal_walking_height
        current_stand_mask = self.is_standing_env[env_ids]
        current_stand_envs_ids = env_ids_tensor[current_stand_mask]
        self.target_height[current_stand_envs_ids] = self._sample_target_height(current_stand_mask.sum())
        self.current_stand_normal_height, self.current_stand_squat_height, self.current_walk = self._update_state(
            self.is_standing_env, self.target_height, self.cfg.squatting_threshold
        )

        # Case 1: previous walk
        if self.prev_walk[env_ids].any():
            # Sub case 1: Current walk: No action
            # Sub case 2: Current stand at normal: no action
            # Sub case 3: Current stand at sqaut: set the squat height to the normal height
            walk_to_stand_mask = self.prev_walk[env_ids] & self.current_stand_squat_height[env_ids]
            if walk_to_stand_mask.any() and not self.random_height_during_walking:
                walk_to_stand_ids = env_ids_tensor[walk_to_stand_mask]
                self.target_height[walk_to_stand_ids] = self.normal_walking_height

        # Case 2: previous stand at normal. All transitions are allowed.

        # Case 3: previous stand at squat.
        if self.prev_stand_squat_height[env_ids].any():
            # Sub case 1: Current stand at normal: no action
            # Sub case 2: Current stand at squat: no action
            # Sub case 3: Current walk: set this env to stand and resample the height
            squat_to_walk_mask = self.prev_stand_squat_height[env_ids] & self.current_walk[env_ids]
            if squat_to_walk_mask.any() and not self.random_height_during_walking:
                squat_to_walk_ids = env_ids_tensor[squat_to_walk_mask]
                self.is_standing_env[squat_to_walk_ids] = True
                self.target_height[squat_to_walk_ids] = self._sample_target_height(squat_to_walk_mask.sum())
                # Update the current state.
                self.current_stand_normal_height, self.current_stand_squat_height, self.current_walk = (
                    self._update_state(self.is_standing_env, self.target_height, self.cfg.squatting_threshold)
                )

        # Make sure all the standing envs have zero velocity command.
        standing_envs_mask = self.is_standing_env[env_ids]
        standing_envs_ids = env_ids_tensor[standing_envs_mask]
        self.vel_command_b[standing_envs_ids, :] = 0.0

        # Check whether we need to randomize the height for walking envs.
        if self.random_height_during_walking:
            current_walk_mask = self.current_walk[env_ids]
            if current_walk_mask.any():
                current_walk_ids = env_ids_tensor[current_walk_mask]
                self.target_height[current_walk_ids] = self._sample_target_height(current_walk_mask.sum())

                crouching_envs = self.target_height[current_walk_ids] < self.cfg.min_walk_height
                if crouching_envs.any():
                    crouching_env_ids = current_walk_ids[crouching_envs]
                    scale = 1 - (self.cfg.min_walk_height - self.target_height[crouching_env_ids]) / (  # type: ignore[call-overload]
                        self.cfg.min_walk_height - self.cfg.ranges.base_height[0]
                    )
                    self.vel_command_b[crouching_env_ids, :] *= scale.unsqueeze(1)  # type: ignore[call-overload]

        # Update tracking for next resample
        self.prev_stand_normal_height = self.current_stand_normal_height
        self.prev_stand_squat_height = self.current_stand_squat_height
        self.prev_walk = self.current_walk

    def _update_metrics(self) -> None:
        super()._update_metrics()
        if self.random_height_during_walking:
            self.error_height_log = torch.abs(self.target_height - self.base_height).abs().mean().item()
        else:
            self.error_height_log = (
                torch.abs(self.target_height - self.base_height)[self.is_standing_env].abs().mean().item()
            )

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame and base height. Shape is (num_envs, 4)."""
        return torch.cat([self.vel_command_b, self.target_height.unsqueeze(-1)], dim=-1)

    def _update_command(self) -> None:
        height = self.robot.data.body_pos_w[:, self._root_id, 2] - self._height_sensor.data.ray_hits_w[..., 2]
        self.base_height = torch.clamp(torch.mean(height, dim=-1), min=0.0, max=5.0)  # clam to prevent inf values

        super()._update_command()

        if not self.random_height_during_walking:
            non_standing_env_ids = (~self.is_standing_env).nonzero(as_tuple=False).flatten()
            self.target_height[non_standing_env_ids] = self.normal_walking_height

    # helpers
    def _update_state(
        self, is_standing_env: torch.Tensor, target_height: torch.Tensor, squat_height_threshold: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Based on the given sampled standing env and target height, compute the state of the robot."""
        current_stand_normal_height = is_standing_env & (target_height >= squat_height_threshold)
        current_stand_squat_height = is_standing_env & (target_height < squat_height_threshold)
        current_walk = ~is_standing_env
        return current_stand_normal_height, current_stand_squat_height, current_walk

    def _sample_target_height(self, num_envs_to_sample: int) -> torch.Tensor:
        target_height = torch.zeros(num_envs_to_sample, device=self.device)
        # Create a 1 to num_envs_to_sample tensor
        local_ids = torch.arange(num_envs_to_sample, device=self.device)
        """Sample the height for the given envs."""
        if self.cfg.bias_height_randomization:
            # Biased sampling
            use_lower = torch.rand(num_envs_to_sample, device=self.device) < self.cfg.lower_height_bias

            if use_lower.any():
                lower_ids = local_ids[use_lower]
                target_height[lower_ids] = torch.empty(use_lower.sum(), device=self.device).uniform_(
                    self.cfg.ranges.base_height[0], self.cfg.sample_middle_height
                )

            if (~use_lower).any():
                upper_ids = local_ids[~use_lower]
                target_height[upper_ids] = torch.empty((~use_lower).sum(), device=self.device).uniform_(
                    self.cfg.sample_middle_height, self.cfg.ranges.base_height[1]
                )
        else:
            # Uniform sampling
            target_height = torch.empty(num_envs_to_sample, device=self.device).uniform_(*self.cfg.ranges.base_height)

        return target_height

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids=env_ids)
        extras["error_height"] = self.error_height_log

        # Reset the previous standing state for the reset envs
        if env_ids is not None:
            self.prev_stand_normal_height[env_ids] = True
            self.prev_stand_squat_height[env_ids] = False
            self.prev_walk[env_ids] = False
        else:
            self.prev_stand_normal_height[:] = True
            self.prev_stand_squat_height[:] = False
            self.prev_walk[:] = False

        return extras


class UniformVelocityGaitBaseHeightCommand(UniformVelocityBaseHeightCommand):
    """Velocity height command with gait phase."""

    cfg: UniformVelocityGaitBaseHeightCommandCfg

    def __init__(self, cfg: UniformVelocityGaitBaseHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.gait_frequency = torch.zeros(env.num_envs, device=self.device)
        self.gait_process = torch.zeros(env.num_envs, device=self.device)
        # the gait process is the time since the cycle started

        self.gait_cycle = torch.zeros(env.num_envs, 2, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        self.gait_frequency[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.gait_frequency_range
        )
        self.gait_process[env_ids] = 0

        # standing envs get frequency 0
        null_velocity = (self.vel_command_b[:, :3] == 0).all(dim=1)
        self.gait_frequency[self.is_standing_env | null_velocity] = 0.0

    def _update_command(self) -> None:
        super()._update_command()
        self.gait_process = torch.fmod(self.gait_process + self._env.step_dt * self.gait_frequency, 1.0)

        self.gait_cycle[:, 0] = torch.sin(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()
        self.gait_cycle[:, 1] = torch.cos(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()
